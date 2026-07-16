use anyhow::Context;
use jlrs::prelude::*;
use jlrs::runtime::handle::local_handle::LocalHandle;
use std::sync::Arc;
use tinned::{
    Expr, ExprTag, ExprVisitor, OneElecMatrix, TinnedError, downcast_from_arc, expression_error,
    generic_error,
};

// Useful for nesting compound expressions, like `Add(a, Mul(b, c))`
struct ExpressionContext {
    tag: ExprTag,
    child_count: usize,
}

// Julia code generator based on `ExprVisitor`
pub struct CodeGenerator {
    julia: LocalHandle,
    expression: String,
    contexts: Vec<ExpressionContext>,
}

impl CodeGenerator {
    pub fn new() -> anyhow::Result<Self> {
        let julia = Builder::new().start_local()?;

        julia.local_scope::<_, 1>(|mut frame| -> JlrsResult<()> {
            unsafe {
                Value::eval_string(&mut frame, "using SpinAdaptedSecondQuantization")?;
            }

            Ok(())
        })?;

        Ok(Self {
            julia,
            expression: String::new(),
            contexts: Vec::new(),
        })
    }

    pub fn print_et_code(&mut self, name: &str) -> anyhow::Result<()> {
        if self.expression.is_empty() {
            anyhow::bail!("Cannot print an empty generated expression");
        }

        if !self.contexts.is_empty() {
            anyhow::bail!("Cannot print an incomplete generated expression");
        }

        let escaped_name = Self::escape_julia_string(name);

        //FIXME: we may call `print_eT_code` to generate Fortran subroutine for eT
        let julia_code =
            format!(r#"print_julia_function("{}", {}) |> print"#, escaped_name, self.expression,);

        self.julia
            .local_scope::<_, 1>(|mut frame| -> JlrsResult<()> {
                unsafe {
                    Value::eval_string(&mut frame, &julia_code)?;
                }

                Ok(())
            })
            .context("failed to evaluate the generated Julia printing code")?;

        Ok(())
    }

    // Before generating a child expression, decide whether the parent needs a
    // separator such as `+`` or `*`, and record that the parent has received
    // another child.
    fn begin_child(&mut self) -> Result<(), TinnedError> {
        let Some(context) = self.contexts.last_mut() else {
            return Ok(());
        };

        // Append the separator only after the first child
        if context.child_count > 0 {
            match context.tag {
                ExprTag::Add => self.expression.push('+'),
                ExprTag::Mul => self.expression.push('*'),
                tag => {
                    return Err(generic_error(
                        format!("Unsupported parent expression {tag:?}"),
                        None,
                    ));
                },
            }
        }

        context.child_count += 1;

        Ok(())
    }

    #[inline]
    fn escape_julia_string(input: &str) -> String {
        input
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t")
    }

    #[inline]
    pub fn reset(&mut self) {
        self.expression.clear();
        self.contexts.clear();
    }
}

impl ExprVisitor for CodeGenerator {
    fn begin(&mut self, tag: ExprTag, _arity: usize) -> Result<(), TinnedError> {
        match tag {
            ExprTag::Add | ExprTag::Mul => {},
            _ => {
                return Err(generic_error(
                    format!("Not implemented for the expression {tag:?}"),
                    None,
                ));
            },
        }

        // The new compound expression is itself a child of its parent.
        self.begin_child()?;

        self.expression.push('(');

        self.contexts.push(ExpressionContext {
            tag,
            child_count: 0,
        });

        Ok(())
    }

    fn leaf(&mut self, tag: ExprTag, expr: &Arc<dyn Expr>) -> Result<(), TinnedError> {
        self.begin_child()?;

        match tag {
            ExprTag::OneElecMatrix => {
                let operator = downcast_from_arc::<OneElecMatrix>(expr)
                    .ok_or_else(|| generic_error("Expected OneElecMatrix", None))?;

                let operator_name = Self::escape_julia_string(operator.name());

                let julia_code =
                    format!(r#"real_tensor("{}", 1, 2) * electron(1, 2)"#, operator_name,);

                self.expression.push_str(&julia_code);
            },

            _ => {
                return Err(expression_error("Not implemented for the expression", expr, None));
            },
        }

        Ok(())
    }

    fn end(&mut self, tag: ExprTag, _arity: usize) -> Result<(), TinnedError> {
        let context = self
            .contexts
            .pop()
            .ok_or_else(|| generic_error(format!("Unexpected end of expression {tag:?}"), None))?;

        if context.tag != tag {
            return Err(generic_error(
                format!("Mismatched expression end: expected {0:?}, received {tag:?}", context.tag),
                None,
            ));
        }

        if context.child_count == 0 {
            return Err(generic_error(format!("Expression {tag:?} has no children"), None));
        }

        self.expression.push(')');

        Ok(())
    }
}
