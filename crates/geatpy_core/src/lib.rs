#![allow(non_snake_case)]

use pyo3::prelude::*;

mod encoding;
mod indicator;
mod multiobjective;
mod mutation;
mod recombination;
mod selection;
pub mod utils;

use encoding::*;
use indicator::*;
use multiobjective::*;
use mutation::*;
use recombination::*;
use selection::*;

#[pymodule(gil_used = false)]
fn _geatpy_core(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // 1. Encoding & Population Generation
    m.add_function(wrap_pyfunction!(crtfld, m)?)?;
    m.add_function(wrap_pyfunction!(bs2ri, m)?)?;
    m.add_function(wrap_pyfunction!(bs2int, m)?)?;
    m.add_function(wrap_pyfunction!(bs2real, m)?)?;
    m.add_function(wrap_pyfunction!(ri2bs, m)?)?;
    m.add_function(wrap_pyfunction!(boundfix, m)?)?;
    m.add_function(wrap_pyfunction!(crtbp, m)?)?;
    m.add_function(wrap_pyfunction!(crtip, m)?)?;
    m.add_function(wrap_pyfunction!(crtrp, m)?)?;
    m.add_function(wrap_pyfunction!(crtpp, m)?)?;
    m.add_function(wrap_pyfunction!(crtri, m)?)?;
    m.add_function(wrap_pyfunction!(crtpc, m)?)?;
    m.add_function(wrap_pyfunction!(crtup, m)?)?;
    m.add_function(wrap_pyfunction!(crtgp, m)?)?;
    m.add_function(wrap_pyfunction!(crtidp, m)?)?;

    // 2. Selection & Ranking
    m.add_function(wrap_pyfunction!(dup, m)?)?;
    m.add_function(wrap_pyfunction!(tour, m)?)?;
    m.add_function(wrap_pyfunction!(etour, m)?)?;
    m.add_function(wrap_pyfunction!(rws, m)?)?;
    m.add_function(wrap_pyfunction!(sus, m)?)?;
    m.add_function(wrap_pyfunction!(urs, m)?)?;
    m.add_function(wrap_pyfunction!(rcs, m)?)?;
    m.add_function(wrap_pyfunction!(rps, m)?)?;
    m.add_function(wrap_pyfunction!(selecting, m)?)?;
    m.add_function(wrap_pyfunction!(mselecting, m)?)?;
    m.add_function(wrap_pyfunction!(ranking, m)?)?;
    m.add_function(wrap_pyfunction!(scaling, m)?)?;
    m.add_function(wrap_pyfunction!(powing, m)?)?;

    // 3. Recombination & Crossover
    m.add_function(wrap_pyfunction!(recsbx, m)?)?;
    m.add_function(wrap_pyfunction!(recdis, m)?)?;
    m.add_function(wrap_pyfunction!(recint, m)?)?;
    m.add_function(wrap_pyfunction!(reclin, m)?)?;
    m.add_function(wrap_pyfunction!(recndx, m)?)?;
    m.add_function(wrap_pyfunction!(xovsp, m)?)?;
    m.add_function(wrap_pyfunction!(xovdp, m)?)?;
    m.add_function(wrap_pyfunction!(xovud, m)?)?;
    m.add_function(wrap_pyfunction!(xovsh, m)?)?;
    m.add_function(wrap_pyfunction!(xovpmx, m)?)?;
    m.add_function(wrap_pyfunction!(xovox, m)?)?;
    m.add_function(wrap_pyfunction!(xovsec, m)?)?;
    m.add_function(wrap_pyfunction!(xovbd, m)?)?;
    m.add_function(wrap_pyfunction!(xovexp, m)?)?;
    m.add_function(wrap_pyfunction!(recombin, m)?)?;

    // 4. Mutation
    m.add_function(wrap_pyfunction!(mutpolyn, m)?)?;
    m.add_function(wrap_pyfunction!(mutgau, m)?)?;
    m.add_function(wrap_pyfunction!(mutbga, m)?)?;
    m.add_function(wrap_pyfunction!(mutbin, m)?)?;
    m.add_function(wrap_pyfunction!(mutde, m)?)?;
    m.add_function(wrap_pyfunction!(mutinv, m)?)?;
    m.add_function(wrap_pyfunction!(mutmove, m)?)?;
    m.add_function(wrap_pyfunction!(mutswap, m)?)?;
    m.add_function(wrap_pyfunction!(mutuni, m)?)?;
    m.add_function(wrap_pyfunction!(mutpp, m)?)?;
    m.add_function(wrap_pyfunction!(mutate, m)?)?;

    // 5. Multi-objective
    m.add_function(wrap_pyfunction!(ndsortESS, m)?)?;
    m.add_function(wrap_pyfunction!(ndsortTNS, m)?)?;
    m.add_function(wrap_pyfunction!(ndsortDED, m)?)?;
    m.add_function(wrap_pyfunction!(crowdis, m)?)?;
    m.add_function(wrap_pyfunction!(cdist, m)?)?;
    m.add_function(wrap_pyfunction!(mergecv, m)?)?;
    m.add_function(wrap_pyfunction!(tcheby, m)?)?;
    m.add_function(wrap_pyfunction!(pbi, m)?)?;
    m.add_function(wrap_pyfunction!(awGA, m)?)?;
    m.add_function(wrap_pyfunction!(rwGA, m)?)?;
    m.add_function(wrap_pyfunction!(refselect, m)?)?;
    m.add_function(wrap_pyfunction!(refgselect, m)?)?;
    m.add_function(wrap_pyfunction!(indexing, m)?)?;
    m.add_function(wrap_pyfunction!(otos, m)?)?;
    m.add_function(wrap_pyfunction!(ecs, m)?)?;
    m.add_function(wrap_pyfunction!(migrate, m)?)?;

    // 6. Indicators (top-level and sub-module)
    m.add_function(wrap_pyfunction!(GD, m)?)?;
    m.add_function(wrap_pyfunction!(IGD, m)?)?;
    m.add_function(wrap_pyfunction!(Spacing, m)?)?;
    m.add_function(wrap_pyfunction!(HV, m)?)?;

    let indicator_submod = PyModule::new(py, "indicator")?;
    indicator_submod.add_function(wrap_pyfunction!(GD, &indicator_submod)?)?;
    indicator_submod.add_function(wrap_pyfunction!(IGD, &indicator_submod)?)?;
    indicator_submod.add_function(wrap_pyfunction!(Spacing, &indicator_submod)?)?;
    indicator_submod.add_function(wrap_pyfunction!(HV, &indicator_submod)?)?;
    m.add_submodule(&indicator_submod)?;

    // Plotting (moeaplot, soeaplot, trcplot, varplot) lives in geatpy/core/_plot.py.

    Ok(())
}
