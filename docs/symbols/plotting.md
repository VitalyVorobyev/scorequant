# Plotting

`scorequant.plotting` is a convenience, not part of the stable API: its figures, layout and
signatures may change in any release. Code that must keep working should draw from the documented
result fields instead. Matplotlib is optional and imported only when one of these functions is
called; install the `viz` extra or synchronize all development extras.

`plot_partition` is a geometric view and therefore accepts only effective rank one or two. For
higher rank, `plot_summary` uses a projection-free retained-eigenvalue spectrum, and
`plot_optimization` summarizes center displacement norms across every informative coordinate.

::: scorequant.plotting.plot_optimization

::: scorequant.plotting.plot_partition

::: scorequant.plotting.plot_information

::: scorequant.plotting.plot_summary
