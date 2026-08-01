"""
ADNeuronSimulation
==================
Backend simulation logic for an interactive 3D neuron viewer that morphs
a neuron model from a healthy state to severe Alzheimer's Disease (AD)
pathology, driven by a single frontend slider value `t` in [0, 100].

This module contains no rendering code -- it is a pure data/math layer.
The frontend (Three.js / WebGL / whatever 3D engine is used) is expected
to call `get_state(t)` (or hit an API endpoint wrapping it) every time the
slider moves, and interpolate mesh deformation, shader/color parameters,
and particle counts (for Aβ plaques, NFTs, microglia) from the returned
values.
"""

import math
import json


class ADNeuronSimulation:
    """
    Simulates the biological progression of a neuron from a healthy state
    to severe Alzheimer's Disease (AD) pathology, driven by a single
    progression parameter `t` in [0, 100].

    -----------------------------------------------------------------
    BIOLOGICAL / MATHEMATICAL RATIONALE
    -----------------------------------------------------------------
    1. Structural INTEGRITY metrics (microtubule integrity, ATP output,
       spine density, soma volume, dendritic branching) are modeled with
       LOGISTIC DECAY curves (inverted S-curves), not straight lines.
       Cells buffer damage via compensatory mechanisms (chaperones,
       mitochondrial reserve capacity, synaptic redundancy) up to a
       tipping point, then fail rapidly once that reserve is exhausted.
       This matches observed tau-driven "microtubule catastrophe" and
       mitochondrial permeability-transition-pore kinetics far better
       than a linear ramp.

    2. Protein AGGREGATION / ACCUMULATION metrics (tau hyperphosphorylation,
       neurofibrillary tangles, amyloid-beta plaques) are modeled with
       LOGISTIC GROWTH curves. Both tau and Aβ misfolding are nucleation-
       dependent, self-templating ("prion-like") processes: a slow lag
       phase while a seed forms, a steep exponential-like growth phase
       once seeding succeeds, then saturation as available substrate
       and physical space are exhausted.

    3. Microglial INFLAMMATION is modeled as a chronic logistic rise
       (long-term "priming"/activation) with a transient Gaussian spike
       layered on top, reflecting the well-documented acute cytokine
       burst microglia mount on first contact with soluble Aβ oligomers,
       before settling into a chronic, less effective, toxic activation
       state in late-stage disease.

    All curves are anchored around the same rough biological timeline
    (Normal: t 0-30, Early AD: t 31-70, Late AD: t 71-100) but are
    deliberately staggered (different midpoints) so that downstream
    events (e.g., NFT accumulation, soma atrophy) visibly lag behind
    their upstream triggers (e.g., tau hyperphosphorylation, microtubule
    loss) -- this mirrors real disease cascades and also gives the 3D
    morph a more convincing, non-synchronized "decay wave" look.
    """

    def __init__(self):
        # Midpoint (t) and steepness (k) for the master logistic curves.
        # midpoint = slider value at which the curve is at its half-way
        # (inflection) point; steepness = how sharply it transitions.
        self.decay_midpoint = 55       # center of structural collapse curves
        self.decay_steepness = 0.12    # how sharply structural metrics fall
        self.growth_midpoint = 50      # center of aggregation growth curves
        self.growth_steepness = 0.13   # how sharply aggregates accumulate

    # -----------------------------------------------------------------
    # Generic curve helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _logistic_growth(t, midpoint, steepness):
        """Standard logistic (S-curve): rises smoothly from 0 -> 1."""
        return 1 / (1 + math.exp(-steepness * (t - midpoint)))

    @classmethod
    def _logistic_decay(cls, t, midpoint, steepness):
        """Inverted logistic: falls smoothly from 1 -> 0. Used for any
        metric that represents structural/functional health declining."""
        return 1 - cls._logistic_growth(t, midpoint, steepness)

    @staticmethod
    def _gaussian_bump(t, center, width, amplitude=1.0):
        """Transient bell-curve spike, used for the acute microglial /
        cytokine burst that occurs early in disease before chronic,
        lower-amplitude inflammation takes over."""
        return amplitude * math.exp(-((t - center) ** 2) / (2 * width ** 2))

    @staticmethod
    def _clamp(value, low=0.0, high=100.0):
        return max(low, min(high, value))

    def _stage_name(self, t):
        """Human-readable disease stage label for UI display."""
        if t <= 30:
            return "Normal"
        elif t <= 70:
            return "Early Stage AD"
        else:
            return "Late Stage / Severe AD"

    # -----------------------------------------------------------------
    # Intracellular metrics
    # -----------------------------------------------------------------

    def calculate_intracellular(self, t):
        """
        Intracellular pathology: microtubule/tau axis, mitochondrial
        function, and ER/Golgi stress.
        """

        # Microtubule integrity: flat near 100% through the Normal window,
        # then a logistic collapse. Tau detachment does not weaken
        # microtubules gradually and evenly -- there is a structural
        # tipping point (catastrophe) once enough tau has dissociated.
        # Midpoint is offset slightly later than tau hyperphosphorylation
        # so integrity loss visibly *follows* tau detachment on the timeline.
        microtubule_integrity = 100 * self._logistic_decay(
            t, midpoint=self.decay_midpoint + 3, steepness=self.decay_steepness
        )

        # Tau hyperphosphorylation: logistic growth. Hyperphosphorylation
        # is a self-propagating, seeded process (kinase dysregulation plus
        # prion-like spreading of misfolded tau between neurons), so it
        # follows nucleation-style kinetics rather than a linear ramp.
        tau_hyperphosphorylation = 100 * self._logistic_growth(
            t, midpoint=self.growth_midpoint, steepness=self.growth_steepness
        )

        # Fraction of tau still correctly bound and stabilizing microtubules
        # -- simplified as the complement of hyperphosphorylated tau.
        tau_bound_stable_pct = self._clamp(100 - tau_hyperphosphorylation)

        # Neurofibrillary tangle (NFT) load: logistic growth, lagging
        # behind tau hyperphosphorylation. NFTs are the downstream
        # aggregate product of free hyperphosphorylated tau, so a pool of
        # detached tau must accumulate before tangles visibly form.
        neurofibrillary_tangle_load = 100 * self._logistic_growth(
            t, midpoint=self.growth_midpoint + 12, steepness=self.growth_steepness
        )

        # Mitochondrial ATP output: starts high (elongated, networked,
        # actively trafficked mitochondria), then falls as fission
        # dominates fusion and transport along microtubules fails.
        # Modeled with logistic decay slightly delayed relative to
        # microtubule loss, reflecting mitochondrial reserve capacity.
        mitochondrial_atp_output = 100 * self._logistic_decay(
            t, midpoint=self.decay_midpoint + 6, steepness=self.decay_steepness
        )
        # Fragmentation/perinuclear pooling is modeled as the complement
        # of healthy networked function.
        mitochondrial_fragmentation_pct = self._clamp(100 - mitochondrial_atp_output)

        # ER/Golgi stress (unfolded protein response activity): logistic
        # growth. Misfolded/aggregating tau and Aβ progressively overwhelm
        # ER chaperone capacity, with an initial compensatory phase
        # (Normal stage) before UPR stress markers climb.
        er_golgi_stress_index = 100 * self._logistic_growth(
            t, midpoint=self.growth_midpoint + 4, steepness=self.decay_steepness
        )

        return {
            "microtubule_integrity_pct": round(microtubule_integrity, 2),
            "tau_bound_stable_pct": round(tau_bound_stable_pct, 2),
            "tau_hyperphosphorylation_pct": round(tau_hyperphosphorylation, 2),
            "neurofibrillary_tangle_load_pct": round(neurofibrillary_tangle_load, 2),
            "mitochondrial_atp_output_pct": round(mitochondrial_atp_output, 2),
            "mitochondrial_fragmentation_pct": round(mitochondrial_fragmentation_pct, 2),
            "er_golgi_stress_index": round(er_golgi_stress_index, 2),
        }

    # -----------------------------------------------------------------
    # Extracellular metrics
    # -----------------------------------------------------------------

    def calculate_extracellular(self, t):
        """
        Extracellular pathology: amyloid-beta dynamics, synaptic loss,
        and the microglial inflammatory response.
        """

        # Amyloid-beta plaque volume: classic logistic growth. Aβ
        # aggregation follows nucleation-dependent polymerization kinetics
        # -- a slow lag phase while monomers/oligomers seed a nucleus, a
        # rapid growth phase once seeding succeeds, then a plateau as
        # available peptide and physical plaque space saturate.
        amyloid_plaque_volume_pct = 100 * self._logistic_growth(
            t, midpoint=self.growth_midpoint - 5, steepness=self.growth_steepness
        )

        # Aβ clearance capacity (microglial phagocytosis, astrocytic
        # uptake, perivascular drainage): starts at 100% (normal clearance)
        # and decays roughly as the inverse of plaque growth, since
        # clearance machinery becomes progressively saturated/overwhelmed.
        amyloid_clearance_pct = 100 * self._logistic_decay(
            t, midpoint=self.growth_midpoint - 5, steepness=self.growth_steepness
        )

        # Dendritic spine density: logistic decay. Spines are lost once
        # soluble Aβ oligomers bind synaptic receptors (e.g., NMDA /
        # mGluR5), so decline tracks oligomer burden with a short lag
        # rather than falling in lockstep with total plaque volume.
        dendritic_spine_density_pct = 100 * self._logistic_decay(
            t, midpoint=self.decay_midpoint - 2, steepness=self.decay_steepness
        )

        # Microglial activation: a chronic logistic rise from baseline
        # "surveillance" mode to sustained "activated/dystrophic" mode,
        # PLUS a transient Gaussian spike around early-stage onset --
        # reflecting the acute inflammatory burst microglia mount on
        # first sensing oligomeric Aβ, before settling into a chronic,
        # less effective, toxic activation state.
        chronic_activation = 100 * self._logistic_growth(
            t, midpoint=self.growth_midpoint + 8, steepness=self.decay_steepness
        )
        acute_activation_spike = 40 * self._gaussian_bump(t, center=45, width=6)
        microglial_activation_pct = self._clamp(chronic_activation + acute_activation_spike)

        # Inflammatory cytokine index (e.g., IL-1β, TNF-α, IL-6 composite):
        # same dual-phase shape as microglial activation -- an early sharp
        # spike (acute response to oligomers) superimposed on a chronic,
        # elevated baseline (neuroinflammatory "priming" in late disease).
        chronic_cytokines = 90 * self._logistic_growth(
            t, midpoint=self.growth_midpoint + 10, steepness=self.decay_steepness
        )
        cytokine_spike = 60 * self._gaussian_bump(t, center=42, width=5)
        inflammatory_cytokine_index = self._clamp(chronic_cytokines + cytokine_spike)

        return {
            "amyloid_beta_plaque_volume_pct": round(amyloid_plaque_volume_pct, 2),
            "amyloid_beta_clearance_pct": round(amyloid_clearance_pct, 2),
            "dendritic_spine_density_pct": round(dendritic_spine_density_pct, 2),
            "microglial_activation_pct": round(microglial_activation_pct, 2),
            "inflammatory_cytokine_index": round(inflammatory_cytokine_index, 2),
        }

    # -----------------------------------------------------------------
    # Morphological metrics (drives the actual 3D mesh deformation)
    # -----------------------------------------------------------------

    def calculate_morphology(self, t):
        """
        Gross 3D morphological parameters for the visual model: soma
        volume and dendritic branching complexity. These are the two
        values most directly useful for driving mesh scale/deformation
        in the frontend renderer.
        """

        # Soma volume: stable during the compensatory Normal stage, then
        # atrophies as organelle failure and NFT burden physically and
        # metabolically shrink the cell body. Logistic decay, deliberately
        # lagging behind microtubule collapse, since gross atrophy is a
        # late structural consequence of earlier molecular failure.
        soma_volume_pct = 100 * self._logistic_decay(
            t, midpoint=self.decay_midpoint + 10, steepness=self.decay_steepness
        )

        # Dendritic branching complexity: logistic decay tracking spine
        # loss but slightly delayed -- branches are only pruned back after
        # sustained synaptic dysfunction, not immediately upon first
        # spine loss.
        dendritic_branching_pct = 100 * self._logistic_decay(
            t, midpoint=self.decay_midpoint, steepness=self.decay_steepness
        )

        return {
            "soma_volume_pct": round(soma_volume_pct, 2),
            "dendritic_branching_pct": round(dendritic_branching_pct, 2),
        }

    # -----------------------------------------------------------------
    # Combined state (single entry point for the frontend)
    # -----------------------------------------------------------------

    def get_state(self, t):
        """
        Returns a full, JSON-serializable snapshot of the neuron's
        simulated biological state at progression value `t` (0-100).
        This is the single method the frontend slider handler should call.
        """
        t = self._clamp(t)  # guard against out-of-range slider input

        return {
            "progression_t": round(t, 2),
            "stage": self._stage_name(t),
            "intracellular": self.calculate_intracellular(t),
            "extracellular": self.calculate_extracellular(t),
            "morphology": self.calculate_morphology(t),
        }

    def get_state_json(self, t):
        """Convenience wrapper: returns the state as a JSON string,
        ready to send straight over an API response body."""
        return json.dumps(self.get_state(t), indent=2)


if __name__ == "__main__":
    # Quick manual sanity check across the full slider range.
    sim = ADNeuronSimulation()
    for test_t in [0, 15, 30, 45, 55, 70, 85, 100]:
        print(f"\n--- t = {test_t} ---")
        print(sim.get_state_json(test_t))