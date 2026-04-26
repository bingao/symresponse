#![doc = include_str!("../../README.md")]

//! ## Crate Structure
//!
//! This crate provides multiple implementations of the [`Lagrangian`] trait:
//!
//! - [`LagrangianCc`] -- coupled-cluster (CC) quasienergy formulation
//! - [`LagrangianMcscf`] -- multi-configurational self-consistent field (MCSCF) quasienergy formulation
//! - [`LagrangianOrbCc`] -- orbital-relaxed CC quasienergy formulation
//! - [`LagrangianDao`] -- atomic-orbital density (DAO) matrix-based quasienergy formulation
//!
//! Internal helper logic is kept private.

mod lagrangian_internal;

pub mod lagrangian;
pub mod lagrangian_cc;
pub mod lagrangian_dao;
pub mod lagrangian_mcscf;
pub mod lagrangian_orb_cc;
pub mod types;

/// Core trait for all Lagrangian implementations.
pub use lagrangian::Lagrangian;

/// Coupled-cluster Lagrangian implementation.
pub use lagrangian_cc::LagrangianCc;

/// DAO-based Lagrangian implementation.
pub use lagrangian_dao::{LagrangianDao, SymmetrizeMode};

/// MCSCF Lagrangian implementation.
pub use lagrangian_mcscf::LagrangianMcscf;

/// Orbital-relaxed CC Lagrangian.
pub use lagrangian_orb_cc::LagrangianOrbCc;

/// Optimal response function and residue output data structure.
pub use types::ResponseDetail;
