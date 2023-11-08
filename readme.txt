Julia implementation of the VASP algorithm for the phase retrieval and MIMIO detection problems.

Our implementations based on the GASP code, whose contents can be downloaded from https://github.com/CarloLucibello/GASP.jl.

The authors would like to thank Prof. Carlo Lucibello for many helpful discussions and his sharing of the GASP code.

Primary Contents

./Phase_Retrieval/: Implementations for the phase retrieval application.
	Main_Iter.jl: Per-iteration MSE of GASP, VASP and VASP's SE.
	Main_LAa.jl: Robustness to parameter change of L and alpha.
	Main_LALa.jl: Robustness to parameter change of L and lambda.

./MIMO_Detection/: Implementations for the MIMO detection application.
	Main_Iter.jl: Per-iteration MSE of AMP, VAMP, GASP, VASP and VASP's SE.
	Main_QQplot.jl and Test7.jl: Validation of Assumption of empirical convergence.