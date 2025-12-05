#![deny(unsafe_op_in_unsafe_fn)]

mod lagrangian;
mod lagrangian_cc;
mod lagrangian_dao;
//mod lagrangian_mcscf;
//mod types;

#[cfg(all(test, feature = "c-headers"))]
mod header_gen {
    use safer_ffi::headers::{Language, NamingConvention, builder};
    use std::{fs, io, path::PathBuf};

    // Generates include/symresponse.h
    #[test]
    #[safer_ffi::cfg_headers]
    fn generate_c_header() -> io::Result<()> {
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("include");
        fs::create_dir_all(&out_dir)?;
        let out = out_dir.join("symresponse.h");

        builder()
            .with_language(Language::C)
            .with_stable_header(true)
            .with_guard("__SYMRESPONSE_H__")
            .with_naming_convention(NamingConvention::Prefix("symresponse_".into()))
            .to_file(&out)?
            .generate()?;

        eprintln!("Generated header at {}", out.display());
        Ok(())
    }
}
