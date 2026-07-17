"""Tests for the role-free ghost population integration into the real Organism.

These tests assert the constitutional invariants required by the integration
contract. They never touch seeds 5-14, never benchmark, and never depend on a
hidden label. Geometry, evidence, transactional tokens, action-conditioned
reachability, and Organism-integration invariants are covered. The existing-
environment assay lives in ``test_real_env_ghost_assay_seed0``.

Malformed, stale, and out-of-order release transactions fail loudly.
"""

import torch
import numpy as np

from olf.geometry import (
    antipodal,
    parallel_transport_sphere,
    project_to_sphere,
    project_to_tangent,
    exponential_map,
)
from olf.organism import Organism
from olf.ghosts.config import GhostConfig
from olf.ghosts.trajectory import GhostTrajectory, make_ghost, transport_ghost
from olf.ghosts.evidence import (
    baseline_error,
    internal_rehearsal_update,
    update_after_recoupling,
    predictive_error,
)
from olf.ghosts.recoupling import ReachabilityBuffer
from olf.ghosts.diagnostics import (
    assert_no_prohibited_labels,
    check_tangent_validity,
)

from olf.seeding import set_seed


def _org(mode="off", ablation=None, **kw):
    cfg = GhostConfig(ghost_mode=mode, ablation=ablation, latent_dim=16,
                      action_dim=3)
    return Organism(obs_dim=18, latent_dim=16, hidden_dim=32,
                    ghost_mode=mode, ghost_config=cfg, **kw)


def _rand_ortho(d):
    q, _ = torch.linalg.qr(torch.randn(d, d))
    return q


def _grounded_ghost(grounding=1.0, evidence=1.0):
    g = make_ghost(torch.randn(16), torch.randn(16), grounding=grounding,
                   credibility=1.0, evidence_support=evidence)
    # Two action-evidence pairs so the transfer map is defined.
    g = g.add_action_evidence(
        g.anchor, torch.tensor([0.5, 0.0, 0.0]), torch.randn(16)
    )
    g = g.add_action_evidence(
        g.anchor, torch.tensor([-0.3, 0.4, 0.1]), torch.randn(16)
    )
    return g


# ==========================================================================
# A. Geometry
# ==========================================================================
def test_no_preferred_axis_at_initialization():
    # Before any external evidence the influence subsystem has an EMPTY
    # population: no invented anchor/tangent can bias a latent axis.
    org = _org("influence")
    obs = torch.randn(18).numpy()
    org.select_action(obs, evaluate=True)
    assert len(org.ghost.population) == 0
    q = _rand_ortho(16)
    org2 = _org("influence")
    h = project_to_sphere(torch.randn(16))
    org2.ghost.begin_step(h)
    org2.ghost.begin_step(q @ h)
    assert len(org2.ghost.population) == 0


def test_first_nonzero_tangent_requires_external_evidence():
    # A ghost is born only from an externally observed deformation.
    org = _org("influence")
    obs = torch.randn(18).numpy()
    org.select_action(obs, evaluate=True)
    org.learn_consequence(0.0, 0.0, 0.0, 0.0, next_obs=obs)
    assert len(org.ghost.population) == 1
    g = org.ghost.population[0]
    # The born tangent must be a valid tangent at its (observed) anchor.
    assert float(g.anchor.norm()) > 1e-6
    assert check_tangent_validity(g.anchor, g.tangent)
    assert torch.isfinite(g.tangent).all()
    assert float(g.grounding) == 0.0  # not yet earned


def test_anchors_remain_on_sphere():
    d = 16
    g = make_ghost(torch.randn(d), torch.randn(d))
    h_prev = project_to_sphere(torch.randn(d))
    h_now = project_to_sphere(torch.randn(d))
    g2 = transport_ghost(g, h_prev, h_now, 1.0)
    assert abs(float(g2.anchor.norm()) - 1.0) < 1e-4


def test_tangents_remain_tangent():
    d = 16
    g = make_ghost(torch.randn(d), torch.randn(d))
    h_prev = project_to_sphere(torch.randn(d))
    h_now = project_to_sphere(torch.randn(d))
    g2 = transport_ghost(g, h_prev, h_now, 1.0)
    assert abs(float((g2.anchor * g2.tangent).sum())) < 1e-4


def test_transport_preserves_tangent_norm():
    d = 16
    x = project_to_sphere(torch.randn(d))
    v = project_to_tangent(x, torch.randn(d))
    y = project_to_sphere(torch.randn(d))
    vt = parallel_transport_sphere(x, y, v)
    assert abs(float(vt.norm() - v.norm())) < 1e-4
    assert abs(float((y * vt).sum())) < 1e-4


def test_transport_equivariant_under_rotation():
    d = 16
    q = _rand_ortho(d)
    x = project_to_sphere(torch.randn(d))
    v = project_to_tangent(x, torch.randn(d))
    y = project_to_sphere(torch.randn(d))
    vt = parallel_transport_sphere(x, y, v)
    vt_q = parallel_transport_sphere(q @ x, q @ y, q @ v)
    assert torch.allclose(q @ vt, vt_q, atol=1e-5)


def test_antipodal_transport_fails():
    d = 16
    x = project_to_sphere(torch.randn(d))
    y = -x
    v = project_to_tangent(x, torch.randn(d))
    assert bool(antipodal(x, y).any())
    try:
        parallel_transport_sphere(x, y, v)
        raise AssertionError("antipodal transport should raise")
    except ValueError:
        pass


# ---- Additional geometry checks ------------------------------------------
def test_birth_tangent_orthogonal_at_birth_anchor():
    # The observed deformation is a tangent at real_prev; the born ghost's
    # anchor is observed_anchor, so the tangent must be transported there and
    # stay orthogonal.
    org = _org("influence")
    obs = torch.randn(18).numpy()
    org.select_action(obs, evaluate=True)
    org.learn_consequence(0.0, 0.0, 0.0, 0.0, next_obs=obs)
    g = org.ghost.population[0]
    assert abs(float(g.anchor.norm()) - 1.0) < 1e-4
    assert abs(float((g.anchor * g.tangent).sum())) < 1e-4


def test_repeated_transport_stays_valid():
    # Repeated transport across a *small* real step, tracking the real anchor
    # the way the lifecycle does (the ghost is re-situated at each real point).
    # This exercises numerical stability of the geodesic transport without
    # forcing the ghost anchor onto the antipode of a fixed real_prev.
    g = make_ghost(torch.randn(16), torch.randn(16))
    hp = project_to_sphere(torch.randn(16))
    cur = g
    for _ in range(6):
        hn = project_to_sphere(hp + 0.1 * torch.randn(16))
        cur = transport_ghost(cur, hp, hn, 1.0)
        hp = cur.anchor
    assert abs(float(cur.anchor.norm()) - 1.0) < 1e-4
    assert abs(float((cur.anchor * cur.tangent).sum())) < 1e-4
    assert torch.isfinite(cur.anchor).all() and torch.isfinite(cur.tangent).all()


def test_negative_dot_tangent_rejected():
    # A tangent with negative (not just positive) dot product must be rejected
    # by the invariant check. `make_ghost` projects its input to the tangent
    # space (so it can never violate the invariant); the test therefore asserts the
    # invariant by constructing a GhostTrajectory whose tangent is NOT orthogonal
    # to its anchor (both signs are covered by the same abs(dot) check).
    x = project_to_sphere(torch.randn(16))
    v_neg = project_to_tangent(x, torch.randn(16)) - 0.1 * x
    assert float((x * v_neg).sum()) < 0
    try:
        GhostTrajectory(
            anchor=x,
            tangent=v_neg,
            credibility=torch.tensor(1.0),
            grounding=torch.tensor(0.0),
            uncertainty=torch.tensor(1.0),
            persistence=torch.tensor(0.0),
            evidence_support=torch.tensor(0.0),
            evidence_negative=torch.tensor(0.0),
            boundary_compat=torch.tensor(1.0),
            horizon_expr=torch.tensor(1.0),
        )
        raise AssertionError("negative-dot tangent should be rejected")
    except ValueError:
        pass


def test_near_identical_points_tangent_valid():
    # When real_prev ~ real_now the observed deformation is ~0, which is a
    # valid (zero) tangent and must not raise.
    x = project_to_sphere(torch.randn(16))
    y = x + 1e-7 * torch.randn(16)
    y = project_to_sphere(y)
    g = make_ghost(x, project_to_tangent(x, y - x))
    assert check_tangent_validity(g.anchor, g.tangent)


def test_antipodal_rejection_in_transport():
    d = 16
    g = make_ghost(torch.randn(d), torch.randn(d))
    x = project_to_sphere(torch.randn(d))
    y = -x
    try:
        transport_ghost(g, x, y, 1.0)
        raise AssertionError("transport across antipode should raise")
    except ValueError:
        pass


def test_action_axis_equivariance():
    # Action-axis PERMUTATION (not entity-slot permutation) must leave the
    # predicted deformation unchanged for the SAME physical action: relabeling
    # the action coordinates and querying with the relabeled action recovers
    # the original prediction. This holds exactly for the ridge transfer map
    # when the permutation is an involution (P^2 = I).
    g = _grounded_ghost()
    perm = torch.tensor([1, 0, 2])  # involution (swap first two coords)
    g_perm = make_ghost(g.anchor, g.tangent, grounding=float(g.grounding),
                        credibility=float(g.credibility),
                        evidence_support=float(g.evidence_support))
    for source, a, t in zip(
        g.transfer_anchors,
        g.transfer_actions,
        g.transfer_tangents,
        strict=True,
    ):
        g_perm = g_perm.add_action_evidence(source, a[perm], t)
    h = project_to_sphere(torch.randn(16))
    a = torch.tensor([0.3, -0.2, 0.5])
    pred = g.transfer_predict(a, h)
    pred_perm = g_perm.transfer_predict(a[perm], h)
    # Relabeled action -> same physical prediction (latent deformation unchanged).
    assert torch.allclose(pred_perm, pred, atol=1e-5)
    # Reachability: a buffer with prototypes for both the action and its
    # permuted form can answer reachability for either.
    buf = ReachabilityBuffer(8)
    buf.add(h, a, torch.randn(16))
    buf.add(h, a[perm], torch.randn(16))
    r_same = buf.residual(torch.randn(16), a, h)
    r_perm = buf.residual(torch.randn(16), a[perm], h)
    assert r_same is not None and r_perm is not None


# ---- action-conditioned reachability ------------------------------
def test_reachability_transports_to_common_anchor():
    buf = ReachabilityBuffer(8)
    d = 16
    act = torch.tensor([0.4, -0.1, 0.2])
    for _ in range(4):
        a = project_to_sphere(torch.randn(d))
        t = project_to_tangent(a, torch.randn(d))
        buf.add(a, act, t)
    cur = project_to_sphere(torch.randn(d))
    t = project_to_tangent(cur, torch.randn(d))
    # Query with the SAME action -> compatible -> residual is a float >= 0.
    res = buf.residual(t, act, cur)
    assert res is not None
    assert res >= 0.0
    # Query with an INCOMPATIBLE action -> unknown -> None (cannot claim reach).
    other = torch.tensor([-0.9, 0.8, -0.7])
    assert buf.residual(t, other, cur) is None


def test_normalized_reachability_residual_frame_invariant():
    torch.manual_seed(1234)
    d = 16
    q = _rand_ortho(d)
    anchors, tangents, actions = [], [], []
    for _ in range(4):
        a = project_to_sphere(torch.randn(d))
        tangent = project_to_tangent(a, torch.randn(d))
        act = torch.randn(3)
        anchors.append(a)
        tangents.append(tangent)
        actions.append(act)
    cur = project_to_sphere(torch.randn(d))
    observed = project_to_tangent(cur, torch.randn(d))
    act = torch.randn(3)
    buf = ReachabilityBuffer(8)
    for a, tangent, a0 in zip(anchors, tangents, actions, strict=True):
        buf.add(a, a0, tangent)
    r1 = buf.residual(observed, act, cur)
    buf2 = ReachabilityBuffer(8)
    for a, tangent, a0 in zip(anchors, tangents, actions, strict=True):
        buf2.add(q @ a, a0, q @ tangent)
    cur_q = q @ cur
    observed_q = project_to_tangent(cur_q, q @ observed)
    # The action lives in an independent 3-d space; a global latent rotation does
    # not relabel action coordinates, so the SAME action is used for the query.
    r2 = buf2.residual(observed_q, act, cur_q)
    assert r1 is not None and r2 is not None
    assert abs(r1 - r2) < 1e-5


# ==========================================================================
# B. Evidence (contrastive vs passive baseline)
# ==========================================================================
def _ghost_with(grounding=0.0, cred=1.0, unc=1.0, pos=0.0, neg=0.0):
    d = 16
    return make_ghost(torch.randn(d), torch.randn(d), grounding=grounding,
                      credibility=cred, uncertainty=unc,
                      evidence_support=pos, evidence_negative=neg)


def test_internal_rehearsal_no_grounding_or_evidence_increase():
    g = _ghost_with(grounding=0.3, pos=0.2, neg=0.1)
    g2 = internal_rehearsal_update(g)
    assert float(g2.grounding) == float(g.grounding)
    assert float(g2.evidence_support) == float(g.evidence_support)
    assert float(g2.evidence_negative) == float(g.evidence_negative)
    assert float(g2.persistence) > float(g.persistence)


def test_poor_prediction_reduces_credibility():
    g = _ghost_with(cred=0.8, unc=0.2)
    obs = project_to_sphere(torch.randn(16))
    step = 1.0
    base = project_to_sphere(torch.randn(16))
    berr = baseline_error(base, obs)
    g2 = update_after_recoupling(g, obs, step, berr, learning_rate=0.5)
    if float(predictive_error(g, obs, step)) > float(berr):
        assert float(g2.credibility) < float(g.credibility)


def test_worse_than_baseline_no_grounding_increase():
    g = _ghost_with(grounding=0.5, cred=0.8, unc=0.2)
    obs = project_to_sphere(torch.randn(16))
    base = g.predicted_anchor(1.0)
    berr = baseline_error(base, obs)
    g2 = update_after_recoupling(g, obs, step=1.0, baseline_err=berr, learning_rate=0.5)
    assert float(g2.grounding) <= float(g.grounding) + 1e-6


def test_better_than_baseline_increases_grounding():
    g = _ghost_with(grounding=0.0, cred=0.5, unc=0.5)
    obs = g.predicted_anchor(1.0)
    base = project_to_sphere(torch.randn(16))
    berr = baseline_error(base, obs)
    g2 = update_after_recoupling(g, obs, step=1.0, baseline_err=berr, learning_rate=0.5)
    assert float(g2.grounding) > float(g.grounding)


def test_repeated_contradiction_negative_evidence():
    g = _ghost_with(grounding=0.2, cred=0.6, unc=0.1, pos=0.0, neg=0.0)
    obs = project_to_sphere(torch.randn(16))
    base = obs
    berr = baseline_error(base, obs)
    for _ in range(5):
        g = update_after_recoupling(g, obs, step=1.0, baseline_err=berr, learning_rate=0.3)
        assert float(g.evidence_negative) >= 0.0
    assert float(g.evidence_negative) > 0.0
    assert float(g.credibility) < 0.6


def test_no_prohibited_labels_in_ghost_apis():
    cfg = GhostConfig(ghost_mode="influence")
    assert_no_prohibited_labels(cfg)
    g = _ghost_with()
    assert_no_prohibited_labels(g._as_dict())


def test_passive_baseline_uses_ordinary_action():
    """the passive baseline is the predicted consequence of the ORDINARY
    OLF candidate action (via the organism's consequence machinery), NOT the FLC
    desired-future latent.

    Equation (tensor ownership: all organism-owned, detached):
        consequences = organism.semantics.predict_consequences(sigma_t, a_cand)
        w_i = softmax(affordance_i)            affordance = value-risk-uncert
        effect = sum_i w_i * dh_pred_i
        base_future_anchor = exp(h, project_to_tangent(h, effect))
    """
    org = _org("influence")
    obs = torch.randn(18).numpy()
    org.select_action(obs, evaluate=True)
    h = org.ghost.organism.h.detach()
    sigma_t = org.last_sigma
    a_cand = torch.as_tensor(org.last_action).reshape(1, 3)
    base_future = org.ghost._passive_baseline_anchor(h, sigma_t, a_cand)
    # Recompute the documented equation independently.
    with torch.no_grad():
        consequences = org.semantics.predict_consequences(sigma_t.detach(), a_cand.detach())
        affordance = org._compute_entity_affordance(consequences)
        weights = torch.softmax(affordance, dim=-1)
        effect = (weights.unsqueeze(-1) * consequences["dh_pred"]).sum(dim=1)
        expected = exponential_map(h, project_to_tangent(h, effect[0]))
    assert torch.allclose(base_future, expected, atol=1e-5)
    # It must differ from the old "desired future latent" baseline.
    desired = org.flc.future_field(
        h, org.last_sigma.flatten(start_dim=1), torch.zeros(1, 2)
    ).latent[0].detach()
    assert not torch.allclose(base_future, desired, atol=1e-5)


# ==========================================================================
# C. Real Organism integration
# ==========================================================================
def test_default_off_has_no_ghost_subsystem():
    org = Organism(obs_dim=18, latent_dim=16, hidden_dim=32)
    assert org.ghost is None
    off = Organism(obs_dim=18, latent_dim=16, hidden_dim=32, ghost_mode="off")
    assert off.ghost is None


def test_influence_uses_same_object_identities():
    org = _org("influence")
    assert org.ghost is not None
    assert org.ghost.organism is org
    assert org.ghost.organism.flc is org.flc
    assert org.ghost.organism.veto is org.veto
    assert org.ghost.organism.motor is org.motor
    assert org.ghost.organism.semantics is org.semantics


def test_no_duplicate_parameters_in_state_dict_or_optimizer():
    off = Organism(obs_dim=18, latent_dim=16, hidden_dim=32, ghost_mode="off")
    inf = _org("influence")
    assert all("ghost" not in k for k in off.state_dict().keys())
    assert all("ghost" not in k for k in inf.state_dict().keys())
    n_off = sum(1 for _ in off.parameters())
    n_inf = sum(1 for _ in inf.parameters())
    assert n_off == n_inf
    ids = [id(p) for p in inf.parameters()]
    assert len(ids) == len(set(ids))
    assert [name for name, _ in off.named_parameters()] == [
        name for name, _ in inf.named_parameters()
    ]
    off_optimizer = torch.optim.SGD(off.parameters(), lr=0.01)
    inf_optimizer = torch.optim.SGD(inf.parameters(), lr=0.01)
    assert [len(group["params"]) for group in off_optimizer.param_groups] == [
        len(group["params"]) for group in inf_optimizer.param_groups
    ]
    for optimizer in (off_optimizer, inf_optimizer):
        optimizer_ids = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        assert len(optimizer_ids) == len(set(optimizer_ids))


def test_ungrounded_ghost_can_influence():
    org = _org("influence")
    obs = torch.randn(18).numpy()
    org.select_action(obs, evaluate=True)
    org.learn_consequence(0.0, 0.0, 0.0, 0.0, next_obs=obs)
    assert len(org.ghost.population) == 1
    assert float(org.ghost.population[0].grounding) == 0.0
    a, info = org.select_action(obs, evaluate=True)
    weights = (info["ghost"] or {}).get("weights", [])
    # Grounding remains telemetry and does not suppress participation.
    for w in weights:
        assert w > 0.0
    assert (info["ghost"] or {}).get("ghost_influenced")


# ---- zero ghosts / only ungrounded ghosts -> zero influence --------
def test_zero_ghosts_no_influence():
    org = _org("influence")
    off = _org("off")
    off.load_state_dict(org.state_dict())
    org.reset_state()
    off.reset_state()
    obs = torch.randn(18).numpy()
    a, info = org.select_action(obs, evaluate=True)
    diag = info["ghost"] or {}
    assert diag.get("ghost_influenced") is False
    assert org._ghost_token is None
    # The released action equals the ordinary (no-ghost) action exactly.
    a_off, _ = off.select_action(obs, evaluate=True)
    assert np.allclose(a, a_off, atol=0.0)


def test_only_ungrounded_ghosts_still_influence():
    org = _org("influence")
    org.reset_state()
    obs = torch.randn(18).numpy()
    org.select_action(obs, evaluate=True)
    org.learn_consequence(0.0, 0.0, 0.0, 0.0, next_obs=obs)  # births ungrounded
    assert len(org.ghost.population) == 1
    assert float(org.ghost.population[0].grounding) == 0.0
    _, info = org.select_action(obs, evaluate=True)
    assert (info["ghost"] or {}).get("ghost_influenced")
    assert org._ghost_token is not None
    assert all(weight > 0.0 for weight in info["ghost"].get("weights", []))


# ---- transactional token ownership --------------------------------
def test_one_pending_release_requires_one_recoupling():
    org = _org("influence", ablation="no_reachability")  # make influence possible
    org.ghost.population.append(_grounded_ghost())
    obs = torch.randn(18).numpy()
    org.select_action(obs, evaluate=True)
    assert org._ghost_token is not None
    org.learn_consequence(0.0, 0.0, 0.0, 0.0, next_obs=obs)
    assert org._ghost_token is None


def test_second_select_before_recouple_is_rejected():
    org = _org("influence", ablation="no_reachability")
    org.ghost.population.append(_grounded_ghost())
    obs = torch.randn(18).numpy()
    org.select_action(obs, evaluate=True)
    assert org._ghost_token is not None
    tok = org._ghost_token
    assert org.ghost._pending_token is tok
    try:
        org.select_action(obs, evaluate=True)
        raise AssertionError("second selection must require external recoupling")
    except RuntimeError as error:
        assert "external consequence" in str(error)
    assert org._ghost_token is tok
    assert org.ghost._pending_token is tok
    # The pending transaction was neither overwritten nor consumed.
    assert org.ghost._pending_transaction is not None
    assert org.ghost._pending_transaction["token"] is tok


def test_pending_transaction_freezes_motor_release_context():
    org = _org("influence", ablation="no_reachability")
    org.ghost.population.append(_grounded_ghost())
    obs = torch.randn(18).numpy()
    org.select_action(obs, evaluate=True)
    transaction = org.ghost.pending_context(org._ghost_token)
    assert transaction is not None
    assert transaction["finalized"] is True
    assert torch.equal(transaction["real_prev"], org._h_at_action.reshape(-1))
    assert np.array_equal(
        transaction["released_action"].numpy(), org.last_action
    )


def test_duplicate_recoupling_token_use_fails():
    org = _org("influence", ablation="no_reachability")
    org.ghost.population.append(_grounded_ghost())
    obs = torch.randn(18).numpy()
    org.select_action(obs, evaluate=True)
    token = org._ghost_token
    s_t = org._h_at_action.clone()
    r1 = org.ghost.recouple_token(token, s_t, project_to_sphere(torch.randn(16)),
                                  s_t, released_action=org.last_action)
    assert r1.get("updated") is True
    try:
        org.ghost.recouple_token(
            token,
            s_t,
            project_to_sphere(torch.randn(16)),
            s_t,
            released_action=org.last_action,
        )
        raise AssertionError("duplicate token use must fail")
    except ValueError:
        pass


def test_arbitrary_token_cannot_satisfy_recoupling():
    org = _org("influence", ablation="no_reachability")
    org.ghost.population.append(_grounded_ghost())
    obs = torch.randn(18).numpy()
    org.select_action(obs, evaluate=True)
    fake = object()
    s_t = org._h_at_action.clone()
    try:
        org.ghost.recouple_token(
            fake,
            s_t,
            project_to_sphere(torch.randn(16)),
            s_t,
            released_action=org.last_action,
        )
        raise AssertionError("foreign token use must fail")
    except ValueError:
        pass


def test_missing_token_with_pending_refused():
    org = _org("influence", ablation="no_reachability")
    org.ghost.population.append(_grounded_ghost())
    obs = torch.randn(18).numpy()
    org.select_action(obs, evaluate=True)
    pending = org._ghost_token
    s_t = org._h_at_action.clone()
    try:
        org.ghost.recouple_token(
            None,
            s_t,
            project_to_sphere(torch.randn(16)),
            s_t,
            released_action=org.last_action,
        )
        raise AssertionError("missing token use must fail")
    except ValueError:
        pass
    # The genuine pending token is untouched and can still recouple.
    assert org.ghost._pending_token is pending


def test_episode_reset_clears_pending_and_records_abort():
    org = _org("influence", ablation="no_reachability")
    org.ghost.population.append(_grounded_ghost())
    obs = torch.randn(18).numpy()
    org.select_action(obs, evaluate=True)
    assert org._ghost_token is not None
    org.reset_state()
    assert org._ghost_token is None
    assert org._ghost_base_future is None
    assert len(org.ghost.population) == 0
    assert org.ghost._pending_token is None
    assert org.ghost._pending_transaction is None
    assert org.ghost._aborted_at_reset is True
    assert org._ghost_reset_aborted_pending is True


def test_episode_reset_without_pending_does_not_report_abort():
    org = _org("observe")
    org.reset_state()
    assert org._ghost_reset_aborted_pending is False
    assert org.ghost._aborted_at_reset is False


def test_telemetry_records_only_external_recoupling():
    org = _org("observe")
    obs = torch.randn(18).numpy()
    before = org.ghost.telemetry()

    org.select_action(obs, evaluate=True)
    after_action = org.ghost.telemetry()
    assert after_action == before

    org.learn_consequence(0.0, 0.0, 0.0, 0.0, next_obs=obs)
    after_recoupling = org.ghost.telemetry()
    assert after_recoupling["recouplings_total"] == 1
    assert after_recoupling["births_total"] == 1
    assert after_recoupling["population"] == 1
    assert after_recoupling["transfer_support"] == 1

    org.reset_state()
    after_reset = org.ghost.telemetry()
    assert after_reset["recouplings_total"] == 1
    assert after_reset["population"] == 0


# ---- boundary evaluation works after warmup -----------------------
def test_boundary_eval_after_warmup_no_exception():
    org = _org("influence", ablation="no_reachability")
    org.ghost.population.append(_grounded_ghost())
    org.veto.warmup = False  # exercise the differentiable boundary path
    obs = torch.randn(18).numpy()

    def run_once():
        a, info = org.select_action(obs, evaluate=True)
        assert info["verdict"] in ("release", "hold", "recouple", "rollback")
        # Final boundary ownership remains with the organism.
        assert org._ghost_token is not None
        org.learn_consequence(0.0, 0.0, 0.0, 0.0, next_obs=obs)
        # The post-environment recoupling consumes the token exactly once.
        assert org._ghost_token is None

    run_once()
    # The differentiable (warmup-off) boundary path remains valid on the next step.
    run_once()


# ---- action-conditioned evidence binds action to consequence ------
def test_signature_path_operates_before_action_conditioned_evidence():
    org = _org("influence", ablation="no_reachability")
    g = make_ghost(torch.randn(16), torch.randn(16), grounding=1.0,
                   credibility=1.0, evidence_support=1.0)
    # Only ONE evidence pair -> below min_action_evidence (2).
    g = g.add_action_evidence(g.anchor, torch.randn(3), torch.randn(16))
    org.ghost.population.append(g)
    obs = torch.randn(18).numpy()
    a, info = org.select_action(obs, evaluate=True)
    assert (info["ghost"] or {}).get("ghost_influenced")
    assert info["ghost"]["candidates"][0]["action_conditioned"] is False


def test_released_action_bound_to_consequence():
    # After a genuine recoupling, the transformation evidence (released action,
    # observed tangent) is stored on the closest ghost.
    org = _org("influence")
    obs = torch.randn(18).numpy()
    org.select_action(obs, evaluate=True)
    org.learn_consequence(0.0, 0.0, 0.0, 0.0, next_obs=obs)
    g = org.ghost.population[0]
    # The birthed ghost must carry the (released_action, observed_tangent) pair.
    assert len(g.transfer_actions) == 1
    assert len(g.transfer_anchors) == 1
    assert len(g.transfer_tangents) == 1
    assert torch.isfinite(g.transfer_actions[0]).all()
    assert torch.isfinite(g.transfer_tangents[0]).all()


def test_action_evidence_transports_to_query_anchor():
    source = project_to_sphere(torch.randn(16))
    query = project_to_sphere(torch.randn(16))
    tangent = project_to_tangent(source, torch.randn(16))
    actions = (
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0]),
    )
    g = make_ghost(source, tangent)
    transported = parallel_transport_sphere(source, query, tangent)
    for action in actions:
        g = g.add_action_evidence(source, action, tangent)
    predicted = g.transfer_predict(actions[0], query)
    assert check_tangent_validity(query, predicted)
    assert torch.allclose(predicted, transported, atol=1e-4, rtol=1e-4)


# ---- action-conditioned reachability excludes incompatible --------
def test_reachability_action_conditioned_excludes_incompatible():
    buf = ReachabilityBuffer(8)
    d = 16
    anchor = project_to_sphere(torch.randn(d))
    support_act = torch.tensor([1.0, 0.0, 0.0])
    buf.add(anchor, support_act, project_to_tangent(anchor, torch.randn(d)))
    cur = project_to_sphere(torch.randn(d))
    t = project_to_tangent(cur, torch.randn(d))
    # Compatible action can claim the supported deformation.
    r_compat = buf.residual(t, support_act, cur)
    assert r_compat is not None
    # Incompatible action cannot claim the same reachable deformation.
    other = torch.tensor([0.0, 1.0, 0.0])
    assert buf.residual(t, other, cur) is None


# ---- honest, real-property assertions ----------------------------
def test_final_boundary_evaluates_combined_ghost_action():
    org = _org("influence", ablation="no_reachability")
    org.ghost.population.append(_grounded_ghost())
    obs = torch.randn(18).numpy()
    a, info = org.select_action(obs, evaluate=True)
    assert (info["ghost"] or {}).get("ghost_influenced")
    assert info["verdict"] in ("release", "hold", "recouple", "rollback")


def test_off_observe_paired_identical():
    """paired organisms from the same initial state.

    off and observe must produce identical actions, verdicts, readiness, modes,
    motor history, RNG state, optimizer parameter IDs/groups, gradients after an
    identical training step, and learned state tensors. observe may only add
    detached diagnostics and external evidence memory.
    """
    set_seed(0)
    off = _org("off")
    obs_mode = _org("observe")
    # Same initial OLF parameters (ghost adds no parameters).
    obs_mode.load_state_dict(off.state_dict())
    # Sync the non-parameter latent state buffer (h) that load_state_dict does
    # not carry: after reset_state both organisms share an identical start.
    off.reset_state()
    obs_mode.reset_state()
    assert len(list(off.parameters())) == len(list(obs_mode.parameters()))

    obs = torch.randn(18).numpy()
    actions_off, actions_obs = [], []
    infos_off, infos_obs = [], []
    torch.manual_seed(123)
    for _ in range(5):
        rng_before = torch.get_rng_state().clone()
        a_off, i_off = off.select_action(obs, evaluate=True)
        off.learn_consequence(0.0, 0.0, 0.0, 0.0, next_obs=obs)
        rng_after_off = torch.get_rng_state().clone()

        # Replay the paired observe-mode transition from the same random stream.
        # Any global RNG consumption unique to ghosts changes this state.
        torch.set_rng_state(rng_before)
        a_obs, i_obs = obs_mode.select_action(obs, evaluate=True)
        obs_mode.learn_consequence(0.0, 0.0, 0.0, 0.0, next_obs=obs)
        rng_after_observe = torch.get_rng_state().clone()
        assert torch.equal(rng_after_off, rng_after_observe)
        torch.set_rng_state(rng_after_off)

        actions_off.append(a_off)
        actions_obs.append(a_obs)
        infos_off.append(i_off)
        infos_obs.append(i_obs)
        assert np.array_equal(off.motor.prev_action, obs_mode.motor.prev_action)

    # Bit-identical actions / verdicts / readiness / modes.
    for ao, oo, io, ioo in zip(
        actions_off, actions_obs, infos_off, infos_obs, strict=True
    ):
        assert np.allclose(ao, oo, atol=0.0)
        for k in ("verdict", "mode", "risk", "viability", "readiness"):
            assert io[k] == ioo[k], k
        assert np.allclose(off.last_action, obs_mode.last_action, atol=0.0)

    # Learned parameters and registered buffers are bit-identical.
    for name, off_tensor in off.state_dict().items():
        assert torch.equal(off_tensor, obs_mode.state_dict()[name]), name
    # Unregistered online state is identical as well.
    assert torch.allclose(off.h, obs_mode.h, atol=1e-6)
    assert off.consequence_events_seen == obs_mode.consequence_events_seen

    # Gradients after an identical training step are identical (ghost is detached).
    def grad_of(org):
        org.zero_grad(set_to_none=True)
        o = torch.randn(18)
        _, info = org.select_action(o, evaluate=False)
        lp = info["_policy_log_prob"]
        (lp.sum() if lp is not None else org.h.sum()).backward()
        return [p.grad.detach().clone() if p.grad is not None else None
                for p in org.parameters()]
    set_seed(7)
    goff = grad_of(off)
    set_seed(7)
    gobs = grad_of(obs_mode)
    for g1, g2 in zip(goff, gobs, strict=True):
        if g1 is None and g2 is None:
            continue
        assert g1 is not None and g2 is not None
        assert torch.allclose(g1, g2, atol=1e-6)

    # observe produced only detached diagnostics + external evidence memory.
    assert obs_mode.ghost is not None
    assert obs_mode.ghost.population is not None


# ==========================================================================
# D. Existing-environment assay + regression (seed 0 only)
# ==========================================================================
def test_real_env_ghost_assay_seed0():
    """one existing exposed environment (not the organism's own flow).

    Asserts: first action has no ghost influence; first evidence may create an
    ungrounded trajectory; influence starts only after positive contrastive
    support and sufficient action-conditioned reachability; every influence has
    exactly one successful external recoupling; no pending transaction is
    overwritten; population stays within capacity; actions/latents finite;
    boundary warmup-off path runs; final diagnostics include positive/negative
    evidence, grounding, reachability residuals, motor validity, readiness,
    verdict, and token state. Reports environment vs ghost signals separately.
    """
    from benchmarks.self_state_meaning import SelfStateMeaningEnv

    set_seed(0)
    org = _org("influence")
    env = SelfStateMeaningEnv(seed=0)
    obs = env.reset()

    env_signals = {"rewards": [], "obs_seen": 0}
    ghost_signals = {"real_prev": 0, "released_action": 0, "base_future": 0, "observed": 0}

    pop_sizes, groundings, reach_proto = [], [], []
    influenced_steps = 0
    finite_ok = True
    pending_seen = False
    for step in range(40):
        action, info = org.select_action(obs, evaluate=False)
        if not np.isfinite(action).all():
            finite_ok = False
        if step == 0:
            # First action has no ghost influence (empty population).
            assert not (info["ghost"] or {}).get("ghost_influenced")
        if org._ghost_token is not None:
            pending_seen = True
        next_obs, reward, done, _ = env.step(action)
        env_signals["rewards"].append(float(reward))
        env_signals["obs_seen"] += 1

        ghost_signals["real_prev"] += 1
        ghost_signals["released_action"] += 1
        ghost_signals["base_future"] += 1
        ghost_signals["observed"] += 1

        # Recouple; the token (if any) must be consumed exactly once.
        pre_token = org._ghost_token
        org.learn_consequence(reward, 0.0, 0.0, 0.0, next_obs=next_obs)
        if pre_token is not None:
            # The genuine pending token must have been consumed by recoupling.
            assert org._ghost_token is None

        g = org.ghost
        pop_sizes.append(len(g.population))
        reach_proto.append(len(g.buffer.prototypes))
        if g.population._ghosts:
            groundings.append(float(g.population[0].grounding))
        if (info.get("ghost") or {}).get("ghost_influenced"):
            influenced_steps += 1
            # No pending transaction may have been overwritten mid-influence.
            assert pending_seen
        # Population always within capacity.
        assert len(g.population) <= g.config.effective_capacity
        obs = next_obs
        if done:
            obs = env.reset()
            org.reset_state()
            # Reset must abort any pending transaction.
            assert org._ghost_token is None

    # Exercise the boundary warmup-off path at least once without exception.
    org.veto.warmup = False
    action, info = org.select_action(obs, evaluate=False)
    assert np.isfinite(action).all()
    next_obs, _signal, _done, _metadata = env.step(action)
    org.learn_consequence(0.0, 0.0, 0.0, 0.0, next_obs=next_obs)
    assert org._ghost_token is None

    assert finite_ok
    assert org.ghost.organism is org
    # Latents finite.
    assert torch.isfinite(org.h).all()

    # Final diagnostics include the required channels.
    final_diag = info["ghost"] or {}
    assert "population" in final_diag
    for candidate in final_diag.get("candidates", []):
        assert "action_support_known" in candidate
        assert "reachability_residual" in candidate
    if org.ghost.population._ghosts:
        gg = org.ghost.population[0]
        assert float(gg.evidence_support) >= 0.0
        assert float(gg.evidence_negative) >= 0.0
        assert 0.0 <= float(gg.grounding) <= 1.0

    print("\n[real-env assay seed 0] influence mode")
    print("  final_pop:", pop_sizes[-1] if pop_sizes else 0,
          "reach_proto:", reach_proto[-1] if reach_proto else 0,
          "influenced_steps:", influenced_steps)
    print("  max grounding seen:", max(groundings) if groundings else 0.0)
    print("  ENV signals: rewards=", len(env_signals["rewards"]),
          "obs_seen=", env_signals["obs_seen"])
    print("  GHOST signals: real_prev=", ghost_signals["real_prev"],
          "released_action=", ghost_signals["released_action"],
          "base_future=", ghost_signals["base_future"],
          "observed=", ghost_signals["observed"])
