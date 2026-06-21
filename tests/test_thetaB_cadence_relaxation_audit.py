from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from tools.diagnostics.thetaB_cadence_relaxation_audit import (
    _classify_report,
    _classify_state_path_report,
    _fixed_theta_row_classification,
    _local_vs_wide_classification,
    render_markdown_report,
    run_audit,
)

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "diagnostics" / "thetaB_cadence_relaxation_audit.py"


def test_thetaB_cadence_relaxation_audit_schema_lists_sections() -> None:
    report = run_audit(mode="schema", protocol=("g1",))

    assert report["meta"]["mode"] == "schema"
    assert report["sections"] == [
        "optimized_trace_replay",
        "fixed_theta_relaxation_matrix",
        "thetaB_scan_sensitivity_matrix",
        "state_path_comparison_matrix",
        "single_step_relaxation_trace",
        "single_pass_outer_survival_trace",
        "thetaB_candidate_state_delta",
        "relaxation_solver_path",
        "full_physics_lane_matrix",
        "full_physics_trace_convergence",
        "full_physics_scaffold_collapse_probe",
        "full_physics_scaffold_support_ownership_probe",
        "trace_continuation_landscape_probe",
        "bending_tilt_out_scaffold_interface_audit",
        "bending_tilt_out_divergence_conditioning_audit",
        "scaffold_geometry_spacing_probe",
        "outer_energy_gradient_assembly",
        "runtime_gradient_bridge",
        "outer_coupling_sign_sweep",
        "base_term_reference_sweep",
        "line_search_interaction",
        "classification",
        "ranked_hypotheses",
    ]
    assert report["defaults"]["trace_passes"] == 12
    assert report["defaults"]["warm_start_policies"] == [
        "anchor_optimized",
        "fresh_fixture",
        "zero_outer",
        "previous_theta",
    ]
    assert report["defaults"]["solver_path_variants"] == [
        "cg_jacobi",
        "cg_none",
        "gd",
    ]
    assert report["defaults"]["assembly_thetas"] == [0.18, 0.21, 0.3]
    assert report["defaults"]["runtime_bridge_thetas"] == [0.18, 0.21, 0.3]
    assert report["defaults"]["full_physics_lane_variants"] == [
        "default_current",
        "full_coupling_trace",
        "full_coupling",
        "ghost",
    ]
    assert report["defaults"]["trace_convergence_epsilons"] == [0.0025, 0.005, 0.01]
    assert report["defaults"]["trace_convergence_geometries"] == [
        "no_trace_current",
        "trace_only",
        "fixed_support",
        "gapfill_release",
    ]
    assert report["defaults"]["scaffold_collapse_geometries"] == [
        "fixed_support",
        "gapfill_release",
    ]
    assert report["defaults"]["scaffold_collapse_variants"] == [
        "plain",
        "projector_only",
        "runtime_options",
        "runtime_options_gd_fallback",
    ]
    assert report["defaults"]["scaffold_support_ownership_variants"] == [
        "runtime_options",
        "runtime_options_gd_fallback",
        "support_tilt_frozen",
        "support_geometry_frozen",
        "support_passive_trace_only",
    ]
    assert report["defaults"]["trace_continuation_landscape_variants"] == [
        "trace_only",
        "fixed_support_runtime_options_gd_fallback",
        "gapfill_release_runtime_options_gd_fallback",
    ]
    assert report["defaults"]["trace_continuation_landscape_modes"] == [
        "trace_tilt",
        "support_tilt",
        "trace_support_tilt",
        "trace_tilt_height",
    ]
    assert report["defaults"]["trace_continuation_landscape_alphas"] == [
        0.25,
        0.5,
        1.0,
    ]
    assert report["defaults"]["bending_tilt_out_interface_reference_modes"] == [
        "current_geometry",
        "flat_reference_zero_J0",
    ]
    assert (
        "fixed_support_eps005_d005"
        in report["defaults"]["scaffold_geometry_spacing_variants"]
    )
    assert (
        "gapfill_release_eps005_n1_trace_reconstructed"
        in report["defaults"]["scaffold_geometry_spacing_variants"]
    )
    assert report["defaults"]["outer_coupling_sweep_thetas"] == [0.18, 0.21, 0.3]
    assert report["defaults"]["outer_coupling_sign_variants"] == [
        "production_sign",
        "flipped_div_sign",
        "flipped_pullback_only",
        "production_sign_current_geometry",
        "production_sign_flat_reference",
    ]
    assert report["defaults"]["base_term_reference_variants"] == [
        "global_flat_reference",
        "inner_flat_outer_current",
        "global_current_geometry",
    ]


def test_thetaB_cadence_classification_helpers_cover_core_cases() -> None:
    assert (
        _local_vs_wide_classification(
            local_best_theta=0.18,
            wide_best_theta=0.24,
            local_best_energy=-1.0,
            wide_best_energy=-1.2,
        )
        == "local_thetaB_scan_trap"
    )

    assert (
        _fixed_theta_row_classification(
            [
                {
                    "elastic_total_from_breakdown": 1.0,
                    "energy_breakdown": {"tilt_out": 0.2, "bending_tilt_out": 0.1},
                },
                {
                    "elastic_total_from_breakdown": 1.3,
                    "energy_breakdown": {"tilt_out": 0.4, "bending_tilt_out": 0.2},
                },
            ]
        )
        == "under_relaxed"
    )

    classification = _classify_report(
        {
            "thetaB_scan_sensitivity_matrix": {
                "rows": [{"classification": "local_thetaB_scan_trap"}]
            },
            "fixed_theta_relaxation_matrix": {
                "theta_summaries": [{"classification": "outer_canceled_by_inner"}]
            },
            "line_search_interaction": [
                {"thetaB_value": 0.18},
                {"thetaB_value": 0.21},
            ],
        }
    )
    assert classification["local_thetaB_scan_trap"] is True
    assert classification["coupled_relaxation_cancellation"] is True
    assert classification["line_search_reduced_relaxation_interference"] is True


def test_thetaB_cadence_state_path_classification_covers_new_cases() -> None:
    classification = _classify_state_path_report(
        {
            "state_path_comparison_matrix": {
                "rows": [
                    {
                        "warm_start_policy": "fresh_fixture",
                        "requested_thetaB_value": 0.21,
                        "relax_steps": 20,
                        "energy_breakdown": {
                            "tilt_out": 0.12,
                            "bending_tilt_out": 0.08,
                        },
                        "outer_shell_field": {"count": 12, "tilt_out_norm_mean": 0.0},
                        "outer_participation": {
                            "outer_shell_row_count": 12,
                            "outer_shell_free_count": 0,
                        },
                        "leaflet_relaxation_stats": {
                            "active_outer_area_rows": 0,
                            "initial_gradient_norm": 1.0,
                            "final_gradient_norm": 0.95,
                            "stop_reason": "line_search_rejected",
                            "tilt_projection_norm_ref_outer_far": 1.0,
                            "tilt_projection_norm_loss_outer_far": 0.8,
                            "projection_apply_count": 1,
                            "outer_shell_row_count": 12,
                        },
                    },
                    {
                        "warm_start_policy": "anchor_optimized",
                        "requested_thetaB_value": 0.21,
                        "relax_steps": 20,
                        "energy_breakdown": {
                            "tilt_out": 0.01,
                            "bending_tilt_out": 0.0,
                        },
                        "outer_shell_field": {"count": 12, "tilt_out_norm_mean": 0.0},
                        "outer_participation": {
                            "outer_shell_row_count": 12,
                            "outer_shell_free_count": 0,
                        },
                        "leaflet_relaxation_stats": {
                            "active_outer_area_rows": 0,
                            "initial_gradient_norm": 1.0,
                            "final_gradient_norm": 0.95,
                            "stop_reason": "line_search_rejected",
                            "tilt_projection_norm_ref_outer_far": 1.0,
                            "tilt_projection_norm_loss_outer_far": 0.8,
                            "projection_apply_count": 1,
                            "outer_shell_row_count": 12,
                        },
                    },
                ]
            }
        }
    )
    assert classification["fresh_anchor_mismatch"] is True
    assert classification["projection_erases_outer_field"] is True
    assert classification["outer_rows_not_free"] is True
    assert classification["outer_area_suppressed"] is True
    assert classification["gradient_stall"] is True


def test_thetaB_cadence_markdown_report_renders_core_sections() -> None:
    text = render_markdown_report(
        {
            "meta": {"mode": "run", "protocol": ["g1"]},
            "optimized_trace_replay": [
                {
                    "label": "default_current",
                    "thetaB_value": 0.18,
                    "tex_total_ratio": 1.01,
                    "elastic_total_from_breakdown": 1.2,
                    "energy_breakdown": {
                        "tilt_out": 0.05,
                        "bending_tilt_out": 0.01,
                    },
                }
            ],
            "ranked_hypotheses": [
                {
                    "label": "local_thetaB_scan_trap",
                    "evidence": "Local trio disagrees with the wider grid.",
                }
            ],
            "classification": {"local_thetaB_scan_trap": True},
            "fixed_theta_relaxation_matrix": {
                "theta_summaries": [
                    {
                        "theta_label": "0.21",
                        "classification": "under_relaxed",
                        "budget_rows": [
                            {
                                "elastic_total_from_breakdown": 1.3,
                                "tilt_out": 0.4,
                                "bending_tilt_out": 0.2,
                            }
                        ],
                    }
                ]
            },
            "state_path_comparison_matrix": {
                "rows": [
                    {
                        "warm_start_policy": "fresh_fixture",
                        "requested_thetaB_value": 0.21,
                        "relax_steps": 20,
                        "elastic_total_from_breakdown": 1.5,
                        "energy_breakdown": {
                            "tilt_out": 0.4,
                            "bending_tilt_out": 0.2,
                        },
                        "outer_shell_field": {"tilt_out_norm_mean": 0.02},
                        "leaflet_relaxation_stats": {
                            "stop_reason": "converged",
                            "final_gradient_norm": 1.0e-3,
                        },
                    }
                ]
            },
            "thetaB_candidate_state_delta": [
                {
                    "requested_thetaB_value": 0.21,
                    "elastic_total_from_breakdown": 1.5,
                    "candidate_delta": {
                        "tilt_out_norm_mean_delta": 0.01,
                        "tilt_out_radial_mean_delta": 0.02,
                    },
                    "leaflet_relaxation_stats": {"stop_reason": "converged"},
                }
            ],
            "relaxation_solver_path": [
                {
                    "solver_path_label": "cg_jacobi",
                    "requested_thetaB_value": 0.21,
                    "elastic_total_from_breakdown": 1.5,
                    "energy_breakdown": {
                        "tilt_out": 0.4,
                        "bending_tilt_out": 0.2,
                    },
                    "leaflet_relaxation_stats": {
                        "gradient_norms_before_constraints": {
                            "out": {"outer_shell": 0.2}
                        },
                        "gradient_norms_after_constraints": {
                            "out": {"outer_shell": 0.1}
                        },
                        "accepted_update_norms_out": {"outer_shell": 0.05},
                        "preconditioner_mean_inv_out": {"outer_shell": 2.0},
                        "stop_reason": "completed_max_iters",
                    },
                }
            ],
            "full_physics_lane_matrix": [
                {
                    "label": "full_coupling",
                    "model_intent": "full_physics_candidate",
                    "reference_mode": "current_geometry",
                    "thetaB_value": 0.21,
                    "tex_total_ratio": 1.1,
                    "tex_ratio_summary": {
                        "contact_ratio": 1.0,
                        "elastic_ratio": 0.9,
                    },
                    "combined_shell_summary": {"norm": 0.01},
                    "first_shell_update_summary": {"norm": 0.002},
                    "outer_participation": {"outer_shell_free_count": 12},
                    "bending_coupling_summary": {"base_term_outer_shell_mean": 0.03},
                    "leaflet_relaxation_stats": {"stop_reason": "converged"},
                }
            ],
            "full_physics_trace_convergence": {
                "rows": [
                    {
                        "label": "full_coupling_trace_eps0005",
                        "epsilon": 0.005,
                        "geometry": "trace_only",
                        "thetaB_value": 0.3,
                        "direct_t_out": 0.14,
                        "direct_phi": 0.14,
                        "shell_grad_norm": 0.04,
                        "shell_update_norm": 0.0003,
                        "contact_ratio": 1.62,
                        "elastic_ratio": 0.74,
                        "classification": "develops_trace",
                    }
                ],
                "summary": {"classification": "trace_only_robust"},
            },
            "full_physics_scaffold_collapse_probe": {
                "rows": [
                    {
                        "geometry": "gapfill_release",
                        "variant": "runtime_options",
                        "thetaB_value": 0.3,
                        "direct_t_out": 0.14,
                        "direct_phi": 0.14,
                        "shell_grad_norm": 0.04,
                        "shell_update_norm": 0.0003,
                        "projector_mode": "continuity_v2",
                        "has_thetaB_boundary_constraint": False,
                        "thetaB_scan_count": 3,
                        "cg_rejection_fallback": "gd",
                        "cg_fallback_attempted_count": 1,
                        "cg_fallback_accepted_count": 1,
                        "cg_fallback_step_size_last": 0.15,
                        "stop_reason": "completed_max_iters",
                        "classification": "develops_trace",
                    }
                ],
                "summary": {"classification": "runtime_options_rescue"},
            },
            "full_physics_scaffold_support_ownership_probe": {
                "rows": [
                    {
                        "geometry": "gapfill_release",
                        "variant": "support_tilt_frozen",
                        "thetaB_value": 0.3,
                        "direct_t_out": 0.14,
                        "direct_phi": 0.14,
                        "role_field_summary": {
                            "trace": {
                                "tilt_out_radial_mean": 0.14,
                                "phi_to_inner": 0.14,
                            },
                            "support": {
                                "tilt_out_radial_mean": 0.05,
                                "phi_to_inner": 0.04,
                            },
                        },
                        "role_gradient_summary": {
                            "trace": {"norm": 0.01},
                            "support": {"norm": 0.02},
                        },
                        "role_update_summary": {
                            "trace": {"norm": 0.001},
                            "support": {"norm": 0.002},
                        },
                        "support_continuation_probe": {
                            "best_sample": {"tilt_dependent_delta": -0.01}
                        },
                        "stop_reason": "completed_max_iters",
                        "classification": "develops_trace",
                    }
                ],
                "summary": {
                    "classification": "support_ownership_ablation_improves_trace"
                },
            },
            "trace_continuation_landscape_probe": {
                "rows": [
                    {
                        "variant": "gapfill_release_runtime_options_gd_fallback",
                        "geometry": "gapfill_release",
                        "thetaB_value": 0.14,
                        "samples": [
                            {
                                "mode": "trace_tilt",
                                "alpha": 0.25,
                                "trace_radial_before": 0.07,
                                "support_radial_before": 0.008,
                                "tilt_dependent_delta": 0.02,
                                "term_deltas": {
                                    "tilt_out": 0.01,
                                    "bending_tilt_out": 0.002,
                                },
                                "dominant_positive_term": "tilt_out",
                            }
                        ],
                    }
                ],
                "summary": {"most_common_suppressing_term": "tilt_out"},
            },
            "bending_tilt_out_scaffold_interface_audit": {
                "rows": [
                    {
                        "variant": "gapfill_release_runtime_options_gd_fallback",
                        "geometry": "gapfill_release",
                        "thetaB_value": 0.14,
                        "direct_t_out": 0.07,
                        "decompositions": [
                            {
                                "reference_mode": "current_geometry",
                                "roles": {
                                    "trace_touching": {
                                        "triangle_count": 6,
                                        "area": 0.2,
                                        "total_energy": 0.04,
                                        "base_energy": 0.01,
                                        "divergence_energy": 0.02,
                                        "cross_energy": 0.01,
                                        "base_term_mean": 0.03,
                                        "divergence_mean": 0.02,
                                    }
                                },
                            }
                        ],
                    }
                ],
                "summary": {"largest_current_geometry_cross_role": "trace_touching"},
            },
            "bending_tilt_out_divergence_conditioning_audit": {
                "rows": [
                    {
                        "variant": "gapfill_release_runtime_options_gd_fallback",
                        "geometry": "gapfill_release",
                        "thetaB_value": 0.14,
                        "direct_t_out": 0.07,
                        "conditioning": {
                            "roles": {
                                "trace_touching": {
                                    "triangle_count": 6,
                                    "area": {"mean": 0.01},
                                    "min_edge": {"min": 0.005},
                                    "aspect": {"mean": 20.0},
                                    "basis_norm": {"max_abs": 100.0},
                                    "divergence": {
                                        "abs_mean": 1.0,
                                        "max_abs": 2.0,
                                    },
                                    "corner_components_by_row_role": {
                                        "trace": {"mean": -0.5},
                                        "support": {"mean": 0.2},
                                        "disk": {"mean": -0.1},
                                    },
                                }
                            }
                        },
                    }
                ],
                "summary": {
                    "max_divergence_role": "gapfill_release_runtime_options_gd_fallback:trace_touching",
                    "max_basis_norm_role": "gapfill_release_runtime_options_gd_fallback:trace_touching",
                },
            },
            "scaffold_geometry_spacing_probe": {
                "rows": [
                    {
                        "label": "fixed_support_eps005_d005",
                        "geometry": "fixed_support",
                        "interface_divergence_mode": "p1_triangle",
                        "outer_shells": 3,
                        "outer_shells_d": 0.05,
                        "thetaB_value": 0.14,
                        "direct_t_out": 0.07,
                        "energy_breakdown": {"bending_tilt_out": 0.03},
                        "trace_area_mean": 0.001,
                        "trace_min_edge": 0.01,
                        "trace_basis_max": 100.0,
                        "trace_div_abs_mean": 1.0,
                        "support_basis_max": 120.0,
                        "support_div_abs_mean": 0.2,
                        "divergence_ablation": {
                            "summary": {
                                "best_energy_delta_label": "trace_divergence_from_support_mean",
                                "best_energy_delta": -0.5,
                                "best_trace_energy_delta": -0.4,
                            }
                        },
                        "gradient_update_probe": {
                            "combined_gradient": {"trace": {"radial_mean": -0.03}},
                            "first_update": {"trace": {"radial_mean": 0.002}},
                        },
                        "high_trace_seed_replay": {
                            "relaxed_direct_t_out": 0.12,
                        },
                        "high_trace_constraint_projection": {
                            "after_trace_t_out": 0.07,
                            "after_gap_to_phi": 0.0,
                        },
                        "shape_gradient_probe": {
                            "roles": {
                                "trace": {"descent_z_mean": -0.01},
                            }
                        },
                        "shape_gradient_module_probe": {
                            "dominant_trace_downward_module": {
                                "module": "surface",
                                "trace_descent_z_mean": -0.02,
                            },
                            "dominant_trace_upward_module": {
                                "module": "bending",
                                "trace_descent_z_mean": 0.01,
                            },
                        },
                        "high_trace_geometry_seed_probe": {
                            "relaxed_direct_t_out": 0.11,
                        },
                        "high_trace_stage_replay_probe": {
                            "rows": [
                                {
                                    "iteration": 0,
                                    "stage": "seed_projected",
                                    "thetaB_value": 0.15,
                                    "trace_t_out": 0.15,
                                    "trace_t_in": 0.01,
                                    "trace_phi": 0.15,
                                    "energy": 1.2,
                                    "dominant_down": {
                                        "module": "tilt_in",
                                        "trace_descent_z_mean": -0.01,
                                    },
                                    "dominant_up": {
                                        "module": "bending_tilt_out",
                                        "trace_descent_z_mean": 0.02,
                                    },
                                }
                            ]
                        },
                        "branch_access_probe": {
                            "rows": [
                                {
                                    "label": "fresh_optimized",
                                    "thetaB_value": 0.12,
                                    "trace_t_out": 0.06,
                                    "trace_phi": 0.06,
                                    "tilt_descent_trace": {"radial_mean": 0.01},
                                    "first_update_trace": {"radial_mean": 0.001},
                                    "shape_trace": {"descent_z_mean": 0.02},
                                    "shape_dominant_up": {
                                        "module": "bending_tilt_out",
                                        "trace_descent_z_mean": 0.02,
                                    },
                                    "shape_dominant_down": {},
                                }
                            ]
                        },
                        "trace_z_fallback_trial_decomposition_probe": {
                            "samples": [
                                {
                                    "alpha": 0.001,
                                    "constraint_context": "minimize",
                                    "energy_delta": 0.01,
                                    "dominant_positive_delta": {
                                        "module": "surface",
                                        "delta": 0.02,
                                    },
                                    "trace_dz_preserved_ratio": 0.5,
                                    "trace_phi_after": 0.07,
                                    "support_phi_after": 0.01,
                                }
                            ]
                        },
                        "role_constraint_tags": {
                            "trace": {
                                "count": 24,
                                "constraints": {"pin_to_circle": 24},
                            }
                        },
                        "stop_reason": "completed_max_iters",
                    }
                ],
                "summary": {
                    "best_trace_divergence_label": "fixed_support_eps005_d005",
                    "best_direct_t_out_label": "fixed_support_eps005_d005",
                    "best_ablation_label": "fixed_support_eps005_d005",
                    "best_ablation_mode": "trace_divergence_from_support_mean",
                    "best_high_seed_label": "fixed_support_eps005_d005",
                    "best_geometry_seed_label": "fixed_support_eps005_d005",
                },
            },
            "outer_energy_gradient_assembly": [
                {
                    "requested_thetaB_value": 0.21,
                    "tilt_out_module": {
                        "tilt_grad_norm_by_region": {"outer_shell": 0.02},
                        "active_row_weight_mean_outer_shell": 1.0,
                    },
                    "bending_tilt_out_module": {
                        "tilt_grad_norm_by_region": {"outer_shell": 0.01},
                        "triangle_counts": {
                            "kept_touching_outer_shell": 10,
                            "full_touching_outer_shell": 12,
                        },
                        "base_term_outer_shell_mean": 0.0,
                    },
                    "combined_outer_shell_gradient": {"norm": 0.005, "cosine": -1.0},
                }
            ],
            "runtime_gradient_bridge": [
                {
                    "requested_thetaB_value": 0.21,
                    "direct_module_outer_gradient": {
                        "tilt_out_shell_norm": 0.02,
                        "bending_tilt_out_shell_norm": 0.01,
                        "tilt_grad_norm_by_region": {"outer_shell": 0.02},
                        "tilt_vs_bending_cosine": -1.0,
                    },
                    "runtime_aggregated_gradient_before_constraints": {
                        "tilt_grad_norm_by_region": {"outer_shell": 0.02}
                    },
                    "runtime_aggregated_gradient_after_constraints": {
                        "tilt_grad_norm_by_region": {"outer_shell": 0.01}
                    },
                    "accepted_update": {
                        "tilt_grad_norm_by_region": {"outer_shell": 0.005}
                    },
                    "shell_vector_comparison": {
                        "direct_vs_runtime_before_cosine": 1.0,
                        "runtime_before_vs_after_cosine": 0.9,
                        "runtime_after_vs_update_cosine": 0.8,
                    },
                }
            ],
            "outer_coupling_sign_sweep": [
                {
                    "variant_label": "production_sign",
                    "requested_thetaB_value": 0.21,
                    "tilt_shell_summary": {"norm": 0.02},
                    "bending_shell_summary": {"norm": 0.01},
                    "combined_shell_summary": {
                        "norm": 0.005,
                        "radial_norm": 0.004,
                        "tangential_norm": 0.003,
                    },
                    "descent_shell_update_summary": {"radial_norm": 0.004},
                    "tilt_vs_bending_cosine": -0.9,
                    "base_vs_divergence_cosine": 0.7,
                    "bending_coupling_summary": {
                        "base_term_outer_shell_mean": 0.0,
                        "div_eval_outer_shell_mean": 0.03,
                    },
                }
            ],
            "base_term_reference_sweep": [
                {
                    "variant_label": "inner_flat_outer_current",
                    "requested_thetaB_value": 0.21,
                    "outer_shell_base_term_mean": 0.03,
                    "tilt_out_shell_gradient": 0.02,
                    "bending_tilt_out_shell_gradient": 0.01,
                    "combined_outer_shell_gradient": {"norm": 0.009, "cosine": -0.4},
                    "first_accepted_shell_update_norm": 0.004,
                    "energy_breakdown": {"tilt_out": 0.4, "bending_tilt_out": 0.2},
                    "tex_ratio_summary": {
                        "contact_ratio": 1.0,
                        "elastic_ratio": 0.5,
                        "total_ratio": 1.1,
                    },
                }
            ],
        }
    )

    assert "# thetaB Cadence / Relaxation Audit" in text
    assert "| Variant | thetaB | tex ratio |" in text
    assert "`local_thetaB_scan_trap`" in text
    assert "| `0.21` | `under_relaxed` |" in text
    assert "## State Path Summary" in text
    assert "## Candidate Delta" in text
    assert "## Solver Path" in text
    assert "## Full-Physics Lane Matrix" in text
    assert "## Full-Physics Trace Convergence" in text
    assert "## Full-Physics Scaffold Collapse Probe" in text
    assert "## Full-Physics Scaffold Support Ownership" in text
    assert "## Trace Continuation Landscape" in text
    assert "## Bending Tilt Out Scaffold Interface" in text
    assert "## Bending Tilt Out Divergence Conditioning" in text
    assert "## Scaffold Geometry Spacing Probe" in text
    assert "### Trace-Z Fallback Trial Decomposition" in text
    assert "## Gradient Assembly" in text
    assert "## Runtime Gradient Bridge" in text
    assert "## Outer Coupling Sign Sweep" in text
    assert "## Base-Term Reference Sweep" in text


def test_thetaB_cadence_cli_schema_writes_yaml_and_markdown(tmp_path) -> None:
    out_yaml = tmp_path / "cadence.yaml"
    out_md = tmp_path / "cadence.md"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mode",
            "schema",
            "--protocol",
            "g1",
            "--out",
            str(out_yaml),
            "--report-out",
            str(out_md),
        ],
        check=True,
        cwd=str(ROOT),
    )

    report = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    assert report["meta"]["mode"] == "schema"
    assert out_md.read_text(encoding="utf-8").startswith(
        "# thetaB Cadence / Relaxation Audit"
    )
