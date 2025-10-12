use safer_ffi::prelude::*;
use std::sync::Arc;

use symresponse::Lagrangian;

/// An *opaque* handle that C can only pass around
#[derive_ReprC]
#[repr(opaque)]
pub struct LagrangianHandle {
    inner: Arc<dyn Lagrangian>,
}

/// Owned box by C after return
pub type LagrangianBox = repr_c::Box<LagrangianHandle>;

impl LagrangianHandle {
    #[inline]
    pub fn new(lag: Arc<dyn Lagrangian>) -> Self {
        Self {
            inner: lag,
        }
    }

    #[inline]
    pub fn as_ref(&self) -> &dyn Lagrangian {
        &*self.inner
    }

    //#[inline]
    //pub fn clone_arc(&self) -> Arc<dyn Lagrangian> {
    //    Arc::clone(&self.inner)
    //}
}

// Free a Lagrangian (NULL-safe).
#[ffi_export]
pub fn symresponse_lagrangian_free(lag: Option<LagrangianBox>) {
    drop(lag);
}
