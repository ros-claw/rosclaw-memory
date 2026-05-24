"""
Unit tests for ROSClaw-Memory embodied core types and algorithms.

No external DB or LLM required — pure Python tests.
"""

import math
from typing import Tuple

import pytest

from powermem.embodied.types import (
    AffectiveTag,
    IntervalRelation,
    MemoryAction,
    Modality,
    PerceptualSnapshot,
    PhysicalInvariant,
    Pose,
    Quaternion,
    TemporalInterval,
    UncertaintyEstimate,
    UncertaintyType,
    Vec3,
)
from powermem.embodied.memory_atom import MemoryAtom
from powermem.embodied.uncertainty import (
    fuse_uncertainties_ci,
    fuse_uncertainties_kalman,
    propagate_spatial_uncertainty,
    suggest_exploration,
    SpatialUncertainty,
)
from powermem.embodied.surprisal_gate import (
    SurprisalGate,
    ZeroOrderHoldPredictor,
    LinearPredictor,
)


# ---------------------------------------------------------------------------
# Vec3 & Pose
# ---------------------------------------------------------------------------

class TestVec3:
    def test_distance(self):
        a = Vec3(0, 0, 0)
        b = Vec3(3, 4, 0)
        assert a.distance_to(b) == pytest.approx(5.0)

    def test_add_sub(self):
        a = Vec3(1, 2, 3)
        b = Vec3(4, 5, 6)
        assert a + b == Vec3(5, 7, 9)
        assert b - a == Vec3(3, 3, 3)

    def test_serde(self):
        v = Vec3(1.5, -2.0, 3.0)
        d = v.to_dict()
        assert Vec3.from_dict(d) == v


class TestPose:
    def test_serde(self):
        p = Pose(position=Vec3(1, 2, 3), orientation=Quaternion(1, 0, 0, 0))
        d = p.to_dict()
        restored = Pose.from_dict(d)
        assert restored.position == p.position
        assert restored.orientation == p.orientation


# ---------------------------------------------------------------------------
# TemporalInterval & Allen Algebra
# ---------------------------------------------------------------------------

class TestTemporalInterval:
    def test_relation_equals(self):
        a = TemporalInterval(0.0, 10.0)
        b = TemporalInterval(0.0, 10.0)
        assert a.relation_to(b) == IntervalRelation.EQUALS

    def test_relation_before(self):
        a = TemporalInterval(0.0, 5.0)
        b = TemporalInterval(10.0, 15.0)
        assert a.relation_to(b) == IntervalRelation.BEFORE
        assert b.relation_to(a) == IntervalRelation.AFTER

    def test_relation_meets(self):
        a = TemporalInterval(0.0, 5.0)
        b = TemporalInterval(5.0, 10.0)
        assert a.relation_to(b) == IntervalRelation.MEETS
        assert b.relation_to(a) == IntervalRelation.MET_BY

    def test_relation_overlaps(self):
        a = TemporalInterval(0.0, 7.0)
        b = TemporalInterval(5.0, 10.0)
        assert a.relation_to(b) == IntervalRelation.OVERLAPS
        assert b.relation_to(a) == IntervalRelation.OVERLAPPED_BY

    def test_relation_during(self):
        a = TemporalInterval(2.0, 8.0)
        b = TemporalInterval(0.0, 10.0)
        assert a.relation_to(b) == IntervalRelation.DURING
        assert b.relation_to(a) == IntervalRelation.CONTAINS

    def test_relation_starts(self):
        a = TemporalInterval(0.0, 5.0)
        b = TemporalInterval(0.0, 10.0)
        assert a.relation_to(b) == IntervalRelation.STARTS
        assert b.relation_to(a) == IntervalRelation.STARTED_BY

    def test_relation_finishes(self):
        a = TemporalInterval(5.0, 10.0)
        b = TemporalInterval(0.0, 10.0)
        assert a.relation_to(b) == IntervalRelation.FINISHES
        assert b.relation_to(a) == IntervalRelation.FINISHED_BY

    def test_invalid_interval_normalized(self):
        # end < start should be normalized to end = start
        t = TemporalInterval(10.0, 5.0)
        assert t.end_sec == 10.0


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------

class TestUncertainty:
    def test_ci_fusion_two_sources(self):
        a = UncertaintyEstimate(std=0.1, confidence=0.9)
        b = UncertaintyEstimate(std=0.2, confidence=0.8)
        fused = fuse_uncertainties_ci([a, b])
        assert 0.1 <= fused.std <= 0.2
        assert 0.8 <= fused.confidence <= 0.9

    def test_ci_epistemic_promotion(self):
        a = UncertaintyEstimate(type=UncertaintyType.ALEATORIC)
        b = UncertaintyEstimate(type=UncertaintyType.EPISTEMIC)
        fused = fuse_uncertainties_ci([a, b])
        assert fused.type == UncertaintyType.EPISTEMIC

    def test_kalman_fusion_reduces_variance(self):
        a = UncertaintyEstimate(std=1.0, confidence=0.5)
        b = UncertaintyEstimate(std=1.0, confidence=0.5)
        fused = fuse_uncertainties_kalman([a, b])
        # Independent sources: fused std should be < individual std
        assert fused.std < 1.0

    def test_spatial_uncertainty_volume(self):
        su = SpatialUncertainty(
            position=Vec3(0, 0, 0),
            covariance=(0.01, 0, 0, 0.01, 0, 0.01),
        )
        vol = su.volume_3sigma()
        assert vol > 0

    def test_exploration_suggestion(self):
        observations = [
            (Vec3(1, 0, 0), UncertaintyEstimate(type=UncertaintyType.EPISTEMIC, confidence=0.2), "rgb"),
            (Vec3(5, 0, 0), UncertaintyEstimate(type=UncertaintyType.ALEATORIC, confidence=0.9), "rgb"),
        ]
        current = Vec3(0, 0, 0)
        suggestions = suggest_exploration(observations, current, max_range=10.0)
        assert len(suggestions) == 1
        assert suggestions[0].target_position == Vec3(1, 0, 0)


# ---------------------------------------------------------------------------
# MemoryAtom
# ---------------------------------------------------------------------------

class TestMemoryAtom:
    def test_roundtrip_metadata(self):
        atom = MemoryAtom(
            content="red cup on table",
            spatial=Vec3(1.0, 2.0, 0.5),
            temporal=TemporalInterval(0.0, 1.0),
            perceptual=PerceptualSnapshot(
                modality=Modality.RGB,
                feature_vec=(0.1, 0.2, 0.3),
                raw_data_hash="abc123",
            ),
            physical=PhysicalInvariant(entity_id="cup_01", mass_kg=0.3),
            uncertainty=UncertaintyEstimate(std=0.05, confidence=0.95),
            affective=AffectiveTag(salience=0.8, trigger="object_detected"),
            action=MemoryAction.OBSERVE,
        )
        meta = atom.to_metadata()
        restored = MemoryAtom.from_metadata(atom.content, meta)

        assert restored.content == atom.content
        assert restored.spatial == atom.spatial
        assert restored.temporal == atom.temporal
        assert restored.perceptual.modality == atom.perceptual.modality
        assert restored.physical.entity_id == atom.physical.entity_id
        assert restored.uncertainty.confidence == atom.uncertainty.confidence
        assert restored.affective.salience == atom.affective.salience
        assert restored.action == atom.action

    def test_from_observation_factory(self):
        atom = MemoryAtom.from_observation(
            content="wall ahead",
            sensor_pose=Pose(position=Vec3(2.0, 0.0, 0.0)),
            modality=Modality.DEPTH,
            feature_vec=(0.5, 0.5),
            timestamp_sec=10.5,
        )
        assert atom.perceptual.modality == Modality.DEPTH
        assert atom.spatial == Vec3(2.0, 0.0, 0.0)
        assert atom.temporal is not None
        assert atom.temporal.start_sec == 10.5

    def test_compute_voxel_key(self):
        atom = MemoryAtom(spatial=Vec3(0.25, 0.35, 0.05))
        key = atom.compute_voxel_key(voxel_size=0.1)
        assert key is not None
        assert "world" in key

    def test_is_significant(self):
        atom_high = MemoryAtom(affective=AffectiveTag(salience=0.9))
        assert atom_high.is_significant is True

        atom_low = MemoryAtom(affective=AffectiveTag(salience=0.3))
        assert atom_low.is_significant is False

    def test_is_high_uncertainty(self):
        atom = MemoryAtom(uncertainty=UncertaintyEstimate(confidence=0.1))
        assert atom.is_high_uncertainty is True

    def test_content_hash_stable(self):
        a = MemoryAtom(content="hello")
        b = MemoryAtom(content="hello")
        assert a.content_hash == b.content_hash


# ---------------------------------------------------------------------------
# SurprisalGate
# ---------------------------------------------------------------------------

class TestSurprisalGate:
    def test_initial_passes_all(self):
        """初始化阶段（样本不足）所有观测都通过"""
        gate = SurprisalGate(min_samples=5, k_sigma=3.0)
        for i in range(4):
            passed, error, threshold = gate.check("cam", float(i))
            assert passed is True
            assert threshold == float("inf")

    def test_steady_state_blocks_redundant(self):
        """稳定后，与历史一致的观测被过滤"""
        gate = SurprisalGate(min_samples=5, k_sigma=3.0)
        # 喂入恒定值建立统计
        for _ in range(20):
            gate.check("cam", 1.0)

        # 相同值应该被过滤
        passed, error, threshold = gate.check("cam", 1.0)
        assert passed is False
        assert error <= threshold

    def test_detects_anomaly(self):
        """异常值应该通过门控"""
        gate = SurprisalGate(min_samples=5, k_sigma=3.0)
        for _ in range(20):
            gate.check("cam", 1.0)

        passed, error, threshold = gate.check("cam", 100.0)
        assert passed is True
        assert error > threshold

    def test_filter_atom(self):
        gate = SurprisalGate(min_samples=3, k_sigma=2.0)
        atom = MemoryAtom(content="test", perceptual=PerceptualSnapshot(modality=Modality.RGB))

        # 前几次应该通过
        result = gate.filter_atom(atom, 1.0, "rgb")
        assert result is not None

        # 建立稳定统计
        for _ in range(10):
            gate.filter_atom(MemoryAtom(content="x"), 1.0, "rgb")

        # 相同值应该被过滤
        result = gate.filter_atom(atom, 1.0, "rgb")
        assert result is None

    def test_predictor_linear_vs_zoh(self):
        zoh = ZeroOrderHoldPredictor()
        lin = LinearPredictor()

        history = [1.0, 2.0, 3.0]
        assert zoh.predict("x", history) == 3.0
        assert lin.predict("x", history) == pytest.approx(4.0)

    def test_state_persistence(self):
        stored = {}
        gate = SurprisalGate(
            min_samples=5,
            state_store=lambda k, v: stored.update({k: v}),
            state_load=lambda k: stored.get(k),
        )
        for i in range(10):
            gate.check("a", float(i))

        # 重建第二个 gate，共享存储
        gate2 = SurprisalGate(
            min_samples=5,
            state_store=lambda k, v: stored.update({k: v}),
            state_load=lambda k: stored.get(k),
        )
        state = gate2.get_state("a")
        assert state["count"] == 10

    def test_reset(self):
        gate = SurprisalGate()
        gate.check("a", 1.0)
        gate.reset("a")
        state = gate.get_state("a")
        assert state["count"] == 0
