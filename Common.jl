import GSL: sf_log_erfc
import CSV
import ForwardDiff, DiffResults
import ForwardDiff: Dual, value, partials

using BSON
using SpecialFunctions
using DataFrames
using Random
using Statistics
using LinearAlgebra
using ExtractMacro
using ForwardDiff
using Parameters
using JLD2
using Distributions
using Cubature
using QuadGK
using IterTools: product
using Plots
using Dates
using Distributed
using SharedArrays

### Special functions ###
# H_Func(x) = (D = Normal(0, 1); cdf(D, - x))
H_Func(x) = erfc(x / sqrt(2)) / 2

function Df_Push(Epochs, Mse_List, Eps_List, Rho_List, A_D_List, A_F_List, A_H_List, A_Chi_List)
	Mse_Iter = mean(Mse_List, dims = 2)
	Eps_Iter = mean(Eps_List, dims = 2)
	Rho_Iter = mean(Rho_List, dims = 2)
	A_D_Iter = mean(A_D_List, dims = 2)
	A_F_Iter = mean(A_F_List, dims = 2)
	A_H_Iter = mean(A_H_List, dims = 2)
	A_Chi_Iter = mean(A_Chi_List, dims = 2)

	Df = DataFrame(
			Epoch = Int[], Mse = Float64[], Eps = Float64[], Rho = Float64[],
			A_D = Float64[], A_F = Float64[], A_H = Float64[], A_Chi = Float64[]
		)

	for i = 1 : Epochs
		Res = (
				Epoch = i, Mse = Mse_Iter[i, 1], Eps = Eps_Iter[i, 1], Rho = Rho_Iter[i, 1],
				A_D = A_D_Iter[i, 1], A_F = A_F_Iter[i, 1], A_H = A_H_Iter[i, 1], A_Chi = A_Chi_Iter[i, 1]
			)
		push!(Df, Res)
	end

	return Df
end

function Df_Push_New(Epochs, Mse_List, Eps_List, Rho_List,
	A_D_List, A_F_List, A_H_List, A_Chi_List,
	S_v_s_z_List, S_HD_2z_List, S_HF_2z_List,
	S_v_s_x_List, S_HD_1x_List, S_HF_1x_List,
	S_v_p_x_List, S_HD_2x_List, S_HF_2x_List,
	S_v_p_z_List, S_HD_1z_List, S_HF_1z_List)

	Mse_Iter = mean(Mse_List, dims = 2)
	Eps_Iter = mean(Eps_List, dims = 2)
	Rho_Iter = mean(Rho_List, dims = 2)
	A_D_Iter = mean(A_D_List, dims = 2)
	A_F_Iter = mean(A_F_List, dims = 2)
	A_H_Iter = mean(A_H_List, dims = 2)
	A_Chi_Iter = mean(A_Chi_List, dims = 2)

	S_v_s_z_Iter = mean(S_v_s_z_List, dims = 2)
	S_HD_2z_Iter = mean(S_HD_2z_List, dims = 2)
	S_HF_2z_Iter = mean(S_HF_2z_List, dims = 2)

	S_v_s_x_Iter = mean(S_v_s_x_List, dims = 2)
	S_HD_1x_Iter = mean(S_HD_1x_List, dims = 2)
	S_HF_1x_Iter = mean(S_HF_1x_List, dims = 2)

	S_v_p_x_Iter = mean(S_v_p_x_List, dims = 2)
	S_HD_2x_Iter = mean(S_HD_2x_List, dims = 2)
	S_HF_2x_Iter = mean(S_HF_2x_List, dims = 2)

	S_v_p_z_Iter = mean(S_v_p_z_List, dims = 2)
	S_HD_1z_Iter = mean(S_HD_1z_List, dims = 2)
	S_HF_1z_Iter = mean(S_HF_1z_List, dims = 2)

	Df = DataFrame(
			Epoch = Int[], Mse = Float64[], Eps = Float64[], Rho = Float64[],
			A_D = Float64[], A_F = Float64[], A_H = Float64[], A_Chi = Float64[],
			S_v_s_z = Float64[], S_HD_2z = Float64[], S_HF_2z = Float64[],
			S_v_s_x = Float64[], S_HD_1x = Float64[], S_HF_1x = Float64[],
			S_v_p_x = Float64[], S_HD_2x = Float64[], S_HF_2x = Float64[],
			S_v_p_z = Float64[], S_HD_1z = Float64[], S_HF_1z = Float64[]
		)

	for i = 1 : Epochs
		Res = (
				Epoch = i, Mse = Mse_Iter[i, 1], Eps = Eps_Iter[i, 1], Rho = Rho_Iter[i, 1],
				A_D = A_D_Iter[i, 1], A_F = A_F_Iter[i, 1], A_H = A_H_Iter[i, 1], A_Chi = A_Chi_Iter[i, 1],
				S_v_s_z = S_v_s_z_Iter[i, 1], S_HD_2z = S_HD_2z_Iter[i, 1], S_HF_2z = S_HF_2z_Iter[i, 1],			
				S_v_s_x = S_v_s_x_Iter[i, 1], S_HD_1x = S_HD_1x_Iter[i, 1], S_HF_1x = S_HF_1x_Iter[i, 1],			
				S_v_p_x = S_v_p_x_Iter[i, 1], S_HD_2x = S_HD_2x_Iter[i, 1], S_HF_2x = S_HF_2x_Iter[i, 1],			
				S_v_p_z = S_v_p_z_Iter[i, 1], S_HD_1z = S_HD_1z_Iter[i, 1], S_HF_1z = S_HF_1z_Iter[i, 1],			
			)
		push!(Df, Res)
	end

	return Df
end

function Report!(Df, Epoch, Mse, Eps, Rho, A_D, A_F, A_H, A_Chi)
	Res = (
			Epoch = Epoch, Mse = Mse, Eps = Eps, Rho = Rho,
			A_D = A_D, A_F = A_F, A_H = A_H, A_Chi = A_Chi
		)
	push!(Df, Res)
end

function Report_New!(Df, Epoch, Mse, Eps, Rho, A_D, A_F, A_H, A_Chi,
	S_v_s_z, S_HD_2z, S_HF_2z,
	S_v_s_x, S_HD_1x, S_HF_1x,
	S_v_p_x, S_HD_2x, S_HF_2x,
	S_v_p_z, S_HD_1z, S_HF_1z)
	Res = (
			Epoch = Epoch, Mse = Mse, Eps = Eps, Rho = Rho,
			A_D = A_D, A_F = A_F, A_H = A_H, A_Chi = A_Chi,
			S_v_s_z = S_v_s_z, S_HD_2z = S_HD_2z, S_HF_2z = S_HF_2z,
			S_v_s_x = S_v_s_x, S_HD_1x = S_HD_1x, S_HF_1x = S_HF_1x,
			S_v_p_x = S_v_p_x, S_HD_2x = S_HD_2x, S_HF_2x = S_HF_2x,
			S_v_p_z = S_v_p_z, S_HD_1z = S_HD_1z, S_HF_1z = S_HF_1z
		)
	push!(Df, Res)
end

### Phase retrieval specific ###
# Optimal spectral init (Mondelli, Montanari '17)
function Spectral_Init(H::Matrix, y::Vector)
	M, N = size(H)
	a = M / N

	y_M = max.(0, y)

	T = Diagonal((y_M .- 1) ./ (y_M .+ sqrt(2 * a) .- 1))
	D = Symmetric(1 / M * H' * T * H)
	La, Vec = eigen(D)

	La_Max, Idx = findmax(La)
	Vec_Max = Vec[:, Idx]

	Project_Sphere!(Vec_Max)

	return Vec_Max
end

# Project on the hypersphere
function Project_Sphere!(Vec_Max::Vector)
	Vec_Max .*= sqrt(length(Vec_Max)) / norm(Vec_Max)
end

function Init_X(H, y, Prob, Prms)
	@extract Prob: N x0
	@extract Prms: InitX

	if InitX isa Vector
		Res = deepcopy(InitX)
	elseif InitX == :Spectral
		Res = Spectral_Init(H, y)
		Idx = sum(sign.(Res) .== sign.(x0))
		if Idx < 0.5 * N
			Res = - Res
		end
	elseif InitX === :Randn
		Res = randn(N)
	elseif InitX === nothing
		Res = zeros(N)
	end

	@assert length(Res) == N

	return Res
end

function Get_Rx(N, cor)
	index = [1 : N ...]

	Mat = cor.^(abs.(index .- index'))

	F = svd(Mat)

	# Rx = U * sqrt.(S) * V'
	Rx = F.U * Diagonal(sqrt.(F.S)) * F.Vt

	return Rx
end

### Problems ###
struct Str_Problem
	Prior

	a::Float64
	N::Int
	M::Int

	Seed::Int

	Ax_T::Float64
	Cx_T::Float64
	vw_T::Float64

	cor_flag::Bool
	cor::Float64
	eps::Float64

	H::Matrix		# measured vector of size M * N
	x0::Vector		# measured vector of size N * 1
	z::Vector		# measured vector of size M * 1
	w::Vector		# measured vector of size M * 1
	y::Vector		# measured vector of size M * 1
end

Problem_Setting(name::String; kws ...) = name == "GLE" ? GLE(; kws ...) : error("Uknown problem")

### Generilized linear estimation ###
function GLE(; Prior = :BPSK_N,
		N = 0, a = 0, Seed = 0,
		Ax_T = 0,
		Cx_T = 0,
		vw_T = 0,
		cor_flag = 0,
		cor = 0,
		eps = 0
	)

	Seed > 0 && Random.seed!(Seed)

	M = round(Int, N * a)

	if Prior == :BPSK_N
		# rm("./BPSK_N.txt", force = true)
		# io = open("./BPSK_N.txt", "w")

		y = rand(Uniform(0, 1), N)
		# println(io, "y: ", y)

		Idx_y = (y .< 1 / 2)
		# println(io, "Idx_y: ", Idx_y)

		In1 = 2 * H_Func(- Ax_T / sqrt(Cx_T)) * y
		# println(io, "In1: ", In1)

		@. In1 = max(In1, 0 + eps)
		@. In1 = min(In1, 1 - eps)
		# println(io, "In1_New: ", In1)

		Inv_In1 = quantile(Normal(), In1)
		# println(io, "Inv_In1: ", Inv_In1)

		Tmp_1 = sqrt(Cx_T) * Inv_In1 .- Ax_T
		# println(io, "Tmp_1: ", Tmp_1)

		In2 = 2 * H_Func(- Ax_T / sqrt(Cx_T)) * (y .- 1) .+ 1
		# println(io, "In2: ", In2)

		@. In2 = max(In2, 0 + eps)
		@. In2 = min(In2, 1 - eps)
		# println(io, "In2_New: ", In2)

		Inv_In2 = quantile(Normal(), In2)	
		# println(io, "Inv_In2: ", Inv_In2)

		Tmp_2 = sqrt(Cx_T) * Inv_In2 .+ Ax_T
		# println(io, "Tmp_2: ", Tmp_2)

		x0 = Idx_y .* Tmp_1 .+ (1 .- Idx_y) .* Tmp_2
		# println(io, "x0: ", x0)

		# close(io)

		# xsaxas
	end

	H = sqrt(1 / N) * randn(M, N)

	if cor_flag
		Rx = Get_Rx(N, cor)
		H = H * Rx
	end

	# println("x0: ", x0)
	println("Norm-x0: ", norm(x0)^(2) / N)

	z = H * x0
	w = sqrt(vw_T) * randn(M)
	y = z .+ w

	# println("z: ", z)
	println("Norm-z: ", norm(z)^(2) / M)

	# xasxasxas

	return Str_Problem(
			Prior,
			a, N, M,
			Seed,
			Ax_T, Cx_T, vw_T,
			cor_flag, cor, eps,
			H, x0, z, w, y
		)
end

function I_Est_RS(m, v, Ax_T, Cx_T, eps)
	pm = (Ax_T * sign.(m) .- m) ./ (v .+ Cx_T)
	# pm2 = 2 * Dirac(m) .* (Ax_T .- abs.(m)) ./ (v .+ Cx_T) .- 1 ./ (v .+ Cx_T)
	pm2 = - 1 ./ (v .+ Cx_T)

	Tau = - pm2

	# Hm = (
	# 		v .* sign.(m) .* Ax_T .+ Cx_T * m
	# 	) ./ (
	# 		v .+ Cx_T
	# 	)

	Hm = pm .* v .+ m
	Hv = v .- Tau .* v.^(2)
	@. Hv = max(Hv, eps)

	Res = (Hm = Hm, Hv = Hv)
	return Res
end

function Log_Phi(m, v1, v0, L, Ax_F, Cx_F)

	Zp = exp(
			- L * (m - Ax_F)^(2) / (
				2 * (Cx_F + v0 + L * v1)
			)
		) * H_Func(
			- (
				L * v1 * Ax_F + (Cx_F + v0) * m
			) / sqrt(
				v1 * (Cx_F + v0) * (Cx_F + v0 + L * v1)
			)
		)

	Zs = exp(
			- L * (m + Ax_F)^(2) / (
				2 * (Cx_F + v0 + L * v1)
			)
		) * H_Func(
			- (
				L * v1 * Ax_F - (Cx_F + v0) * m
			) / sqrt(
				v1 * (Cx_F + v0) * (Cx_F + v0 + L * v1)
			)
		)

	# lp = - 1 / 2 * log(v0 + Cx_F) -
	# 	log(
	# 		H_Func(- Ax_F / sqrt(v0 + Cx_F))
	# 	) -
	# 	1 / (2 * L) * log(v0 + Cx_F + L * v1) +
	# 	1 / (2 * L) * log(v0 + Cx_F) +
	# 	1 / L * log(Zs + Zp)

	lp = 1 / (2 * L) * log(Cx_F + v0) -
		1 / (2 * L) * log(Cx_F + v0 + L * v1) +
		1 / L * log(Zp + Zs)

	return lp
end

### Use the derivative from replica computations ###
P_m(m, v1, v0, L, Ax_F, Cx_F) = ForwardDiff.derivative(m -> Log_Phi(m, v1, v0, L, Ax_F, Cx_F), m)
P_v1(m, v1, v0, L, Ax_F, Cx_F) = ForwardDiff.derivative(v1 -> Log_Phi(m, v1, v0, L, Ax_F, Cx_F), v1)
P_v0(m, v1, v0, L, Ax_F, Cx_F) = ForwardDiff.derivative(v0 -> Log_Phi(m, v1, v0, L, Ax_F, Cx_F), v0)

function I_Est_1RSB(m, v1, v0, L, Ax_F, Cx_F, eps)
	v = v0 .+ L * v1

	pm = @. P_m(m, v1, v0, L, Ax_F, Cx_F)
	pv1 = @. P_v1(m, v1, v0, L, Ax_F, Cx_F)
	pv0 = @. P_v0(m, v1, v0, L, Ax_F, Cx_F)
	# pm2 = - (- 2 * pv1 .+ L * pm.^(2))

	# Tau0 = 2 / (L - 1) * (pv1 .- L * pv0)
	Tau0 = - 2 * (pv1 .- L * pv0)

	# Tau = - pm2
	Tau = - 2 * pv1 .+ L * abs.(pm).^(2)

	Hm = pm .* v .+ m

	Hv0 = v0 .- Tau0 .* abs.(v0).^(2)
	@. Hv0 = max(Hv0, eps)

	Hv = v .- Tau .* abs.(v).^(2)
	@. Hv = max(Hv, Hv0 + eps)

	Hv1 = 1 / L * (Hv .- Hv0)
	@. Hv1 = max(Hv1, eps)

	Res = (Hm = Hm, Hv1 = Hv1, Hv0 = Hv0)

	return Res
end

function O_Est_RS(y, m, v, vw_T, eps)
	Hv = 1 ./ (1 ./ v .+ 1 / vw_T)
	@. Hv = max(Hv, eps)

	Hm = Hv .* (m ./ v .+ y / vw_T)

	Res = (Hm = Hm, Hv = Hv)
	return Res
end

function O_Est_1RSB(y, m, v1, v0, L, vw_F, eps)
	v = v0 .+ L * v1

	Hv0 = 1 ./ (1 ./ v0 .+ 1 / vw_F)
	@. Hv0 = max(Hv0, eps)

	Hv = 1 ./ (1 ./ v .+ 1 / vw_F)
	@. Hv = max(Hv, Hv0 .+ eps)

	Hv1 = 1 / L * (Hv .- Hv0)
	@. Hv1 = max(Hv1, eps)

	Hm = Hv .* (m ./ v .+ y / vw_F)

	Res = (Hm = Hm, Hv1 = Hv1, Hv0 = Hv0)

	return Res
end

### Parameters for solve ###
@with_kw struct Str_RS_Params
	Epochs = 0
	Sd = 0
	InitX = :Spectral	# initial configuration [a configuration, :Spectral, :Randn, nothing]
	Epsilon = 0			# stopping criterion
	Mes = 0				# damping rate
	Ax_T = 0
	Cx_T = 0
	vw_T = 0			# the variance of likelihood function
	eps = 0
	Verb = 0
	Name = "Default"
end

@with_kw struct Str_1RSB_Params
	Epochs = 0
	L = 0				# parisi parameter
	Sd = 0
	InitX = :Spectral	# initial configuration [a configuration, :Spectral, :Randn, nothing]
	Epsilon = 0			# stopping criterion
	Mes = 0				# damping rate
	Ax_F = 0
	Cx_F = 0
	vw_F = 0			# the variance of likelihood function
	eps = 0
	Verb = 0
	Name = "Default"
end

function Pri_IO(io, str, vec)
	println(io, str, [maximum(vec), minimum(vec), mean(vec)])
end

function Gauss(x, m, v)
	Res = 1 / sqrt(2 * pi * v) * exp(
			- 1 / (2 * v) * (x - m)^(2)
		)

	return Res
end

function Second_Integral_Base(Ax_T, Cx_T, Func, Min_X::Vector, Max_X::Vector)
	Res = hcubature(
			W_X -> begin
				w = W_X[1]
				x = W_X[2]

				tmp = Gauss(w, 0, 1) * Gauss(abs(x), Ax_T, Cx_T) * Func(w, x)
				isfinite(tmp) ? tmp : 0.
			end,
			# Min_X, Max_X, abstol = 1e-8
			# Min_X, Max_X, abstol = 1e-7 best
			Min_X, Max_X, reltol = 1e-8, abstol = 1e-9, maxevals = 0
		)

	return Res[1]
end

function Second_Integral(Ax_T, Cx_T, Func)
	dx = 1e-2
	# dist = 40
	dist = 20
	# dist = 10

	List = map(
			x -> sign(x) * abs(x)^(2), -1 : dx : 1
		) * dist

	Ints = [
			(List[i], List[i + 1]) for i = 1 : length(List) - 1
		]
	Intprods = product(Ints, Ints)

	Res = sum(
		Ip -> begin
			Min_X = [Ip[1][1], Ip[2][1]]
			Max_X = [Ip[1][2], Ip[2][2]]
			Second_Integral_Base(Ax_T, Cx_T, Func, Min_X, Max_X)
		end, Intprods
	)

	return Res
end

function Second_Integral_Base_NP(Ax_T, Func, Min_X::Vector, Max_X::Vector)
	Res = hcubature(
			W_N -> begin
				w = W_N[1]
				n = W_N[2]

				tmp = Gauss(w, 0, 1) * Gauss(n, 0, 1) * Func(w, n)
				isfinite(tmp) ? tmp : 0.
			end,
			# Min_X, Max_X, abstol = 1e-8
			# Min_X, Max_X, abstol = 1e-7 best
			Min_X, Max_X, reltol = 1e-8, abstol = 1e-9, maxevals = 0
		)

	return Res[1]
end

function Second_Integral_N(Ax_T, Cx_T, Func)
	dx = 1e-2
	# dist = 40
	dist = 20
	# dist = 10

	List_1 = map(
			x -> sign(x) * abs(x)^(2), -1 : dx : 1
		) * dist

	Ints_1 = [
			(List_1[i], List_1[i + 1]) for i = 1 : length(List_1) - 1
		]

	Size = size(List_1, 1)

	List_2 = [List_1[1]]
	
	for i = 2 : Size
		if List_1[i] > Ax_T / sqrt(Cx_T)
			List_2 = [List_2 Ax_T / sqrt(Cx_T)]
			break
		else
			List_2 = [List_2 List_1[i]]
		end
	end

	Ints_2 = [
			(List_2[i], List_2[i + 1]) for i = 1 : length(List_2) - 1
		]

	Intprods = product(Ints_1, Ints_2)

	Res = sum(
		Ip -> begin
			Min_X = [Ip[1][1], Ip[2][1]]
			Max_X = [Ip[1][2], Ip[2][2]]
			Second_Integral_Base_NP(Ax_T, Func, Min_X, Max_X)
		end, Intprods
	)

	return Res
end

function Second_Integral_P(Ax_T, Cx_T, Func)
	dx = 1e-2
	# dist = 40
	dist = 20
	# dist = 10

	List_1 = map(
			x -> sign(x) * abs(x)^(2), -1 : dx : 1
		) * dist

	Ints_1 = [
			(List_1[i], List_1[i + 1]) for i = 1 : length(List_1) - 1
		]

	Size = size(List_1, 1)

	List_2 = [- List_1[1]]

	for i = 2 : Size
		if List_1[i] > Ax_T / sqrt(Cx_T)
			List_2 = [- Ax_T / sqrt(Cx_T) List_2]
			break
		else
			List_2 = [- List_1[i] List_2]
		end
	end

	Ints_2 = [
			(List_2[i], List_2[i + 1]) for i = 1 : length(List_2) - 1
		]

	Intprods = product(Ints_1, Ints_2)

	Res = sum(
		Ip -> begin
			Min_X = [Ip[1][1], Ip[2][1]]
			Max_X = [Ip[1][2], Ip[2][2]]
			Second_Integral_Base_NP(Ax_T, Func, Min_X, Max_X)
		end, Intprods
	)

	return Res
end

function SE_Hm(m, v1, v0, L, Ax_F, Cx_F)
	v = v0 + L * v1

	pm = P_m(m, v1, v0, L, Ax_F, Cx_F)
	Hm = pm * v + m

	return Hm
end

function SE_Hv1(m, v1, v0, L, Ax_F, Cx_F)
	v = v0 + L * v1

	pm = P_m(m, v1, v0, L, Ax_F, Cx_F)
	pv1 = P_v1(m, v1, v0, L, Ax_F, Cx_F)
	pv0 = P_v0(m, v1, v0, L, Ax_F, Cx_F)

	# Tau0 = 2 / (L - 1) * (pv1 - L * pv0)
	Tau0 = - 2 * (pv1 - L * pv0)
	Tau = - 2 * pv1 + L * pm^(2)

	Hv0 = v0 - Tau0 * v0^(2)
	Hv = v - Tau * v^(2)

	Hv1 = 1 / L * (Hv - Hv0)

	return Hv1
end

function SE_Hv0(m, v1, v0, L, Ax_F, Cx_F)

	pv1 = P_v1(m, v1, v0, L, Ax_F, Cx_F)
	pv0 = P_v0(m, v1, v0, L, Ax_F, Cx_F)

	# Tau0 = 2 / (L - 1) * (pv1 - L * pv0)
	Tau0 = - 2 * (pv1 - L * pv0)

	Hv0 = v0 - Tau0 * v0^(2)

	return Hv0
end

function I_SE_1RSB(HD_1x, HF_1x, v1, v0, L, Ax_T, Cx_T, Ax_F, Cx_F, eps)
	v = v0 + L * v1
	r = 1 / v

	C = 2 * H_Func(- Ax_T / sqrt(Cx_T))
	println("C: ", C)

	DX_1 = 1 / C * Second_Integral(
			Ax_T, Cx_T,
			(w, x) -> begin
				x * SE_Hm(
					HD_1x / r * x +
					sqrt(HF_1x / r^(2)) * w,
					v1, v0, L, Ax_F, Cx_F
				)
			end
		)

	# DX_2_p_m = Second_Integral_P(
	# 		Ax_T, Cx_T,
	# 		(w, n) -> begin
	# 			SE_Hm(
	# 				HD_1x / r * (Ax_T + sqrt(Cx_T) * n) +
	# 				sqrt(HF_1x / r^(2)) * w,
	# 				v1, v0, L, Ax_F, Cx_F
	# 			)
	# 		end
	# 	)
	# DX_2_p_nm = Second_Integral_P(
	# 		Ax_T, Cx_T,
	# 		(w, n) -> begin
	# 			n * SE_Hm(
	# 				HD_1x / r * (Ax_T + sqrt(Cx_T) * n) +
	# 				sqrt(HF_1x / r^(2)) * w,
	# 				v1, v0, L, Ax_F, Cx_F
	# 			)
	# 		end
	# 	)
	# DX_2_n_m = Second_Integral_N(
	# 		Ax_T, Cx_T,
	# 		(w, n) -> begin
	# 			SE_Hm(
	# 				HD_1x / r * (- Ax_T + sqrt(Cx_T) * n) +
	# 				sqrt(HF_1x / r^(2)) * w,
	# 				v1, v0, L, Ax_F, Cx_F
	# 			)
	# 		end
	# 	)
	# DX_2_n_nm = Second_Integral_N(
	# 		Ax_T, Cx_T,
	# 		(w, n) -> begin
	# 			n * SE_Hm(
	# 				HD_1x / r * (- Ax_T + sqrt(Cx_T) * n) +
	# 				sqrt(HF_1x / r^(2)) * w,
	# 				v1, v0, L, Ax_F, Cx_F
	# 			)
	# 		end
	# 	)

	# DX_2 = 1 / C * (
	# 		Ax_T * DX_2_p_m + sqrt(Cx_T) * DX_2_p_nm -
	# 		Ax_T * DX_2_n_m + sqrt(Cx_T) * DX_2_n_nm
	# 	)

	# println("DX_1: ", DX_1)
	# println("DX_2: ", DX_2)

	FX_1 = 1 / C * Second_Integral(
			Ax_T, Cx_T,
			(w, x) -> begin
				(
					SE_Hm(
						HD_1x / r * x +
						sqrt(HF_1x / r^(2)) * w,
						v1, v0, L, Ax_F, Cx_F
					)
				)^(2)
			end
		)

	# FX_2_p = Second_Integral_P(
	# 		Ax_T, Cx_T,
	# 		(w, n) -> begin
	# 			(
	# 				SE_Hm(
	# 					HD_1x / r * (Ax_T + sqrt(Cx_T) * n) +
	# 					sqrt(HF_1x / r^(2)) * w,
	# 					v1, v0, L, Ax_F, Cx_F
	# 				)
	# 			)^(2)
	# 		end
	# 	)
	# FX_2_n = Second_Integral_N(
	# 		Ax_T, Cx_T,
	# 		(w, n) -> begin
	# 			(
	# 				SE_Hm(
	# 					HD_1x / r * (- Ax_T + sqrt(Cx_T) * n) +
	# 					sqrt(HF_1x / r^(2)) * w,
	# 					v1, v0, L, Ax_F, Cx_F
	# 				)
	# 			)^(2)
	# 		end
	# 	)
	# FX_2 = 1 / C * (FX_2_p + FX_2_n)

	# println("FX_1: ", FX_1)
	# println("FX_2: ", FX_2)

	HX_1 = 1 / C * Second_Integral(
			Ax_T, Cx_T,
			(w, x) -> begin
				SE_Hv1(
					HD_1x / r * x +
					sqrt(HF_1x / r^(2)) * w,
					v1, v0, L, Ax_F, Cx_F
				)
			end
		)

	# HX_2_p = Second_Integral_P(
	# 		Ax_T, Cx_T,
	# 		(w, n) -> begin
	# 			SE_Hv1(
	# 				HD_1x / r * (Ax_T + sqrt(Cx_T) * n) +
	# 				sqrt(HF_1x / r^(2)) * w,
	# 				v1, v0, L, Ax_F, Cx_F
	# 			)
	# 		end
	# 	)
	# HX_2_n = Second_Integral_N(
	# 		Ax_T, Cx_T,
	# 		(w, n) -> begin
	# 			SE_Hv1(
	# 				HD_1x / r * (- Ax_T + sqrt(Cx_T) * n) +
	# 				sqrt(HF_1x / r^(2)) * w,
	# 				v1, v0, L, Ax_F, Cx_F
	# 			)
	# 		end
	# 	)
	# HX_2 = 1 / C * (HX_2_p + HX_2_n)

	# println("HX_1: ", HX_1)
	# println("HX_2: ", HX_2)

	ChiX_1 = 1 / C * Second_Integral(
			Ax_T, Cx_T,
			(w, x) -> begin
				SE_Hv0(
					HD_1x / r * x +
					sqrt(HF_1x / r^(2)) * w,
					v1, v0, L, Ax_F, Cx_F
				)
			end
		)

	# ChiX_2_p = Second_Integral_P(
	# 		Ax_T, Cx_T,
	# 		(w, n) -> begin
	# 			SE_Hv0(
	# 				HD_1x / r * (Ax_T + sqrt(Cx_T) * n) +
	# 				sqrt(HF_1x / r^(2)) * w,
	# 				v1, v0, L, Ax_F, Cx_F
	# 			)
	# 		end
	# 	)
	# ChiX_2_n = Second_Integral_N(
	# 		Ax_T, Cx_T,
	# 		(w, n) -> begin
	# 			SE_Hv0(
	# 				HD_1x / r * (- Ax_T + sqrt(Cx_T) * n) +
	# 				sqrt(HF_1x / r^(2)) * w,
	# 				v1, v0, L, Ax_F, Cx_F
	# 			)
	# 		end
	# 	)
	# ChiX_2 = 1 / C * ( ChiX_2_p + ChiX_2_n)

	# println("ChiX_1: ", ChiX_1)
	# println("ChiX_2: ", ChiX_2)

	Res = (DX = DX_1, FX = FX_1, HX = HX_1, ChiX = ChiX_1)
	# Res = (DX = DX_2, FX = FX_2, HX = HX_2, ChiX = ChiX_2)
	return Res
end

function O_SE_1RSB(HC_1z, HD_1z, HF_1z, v1, v0, L, vw_T, vw_F, eps)
	v = v0 + L * v1

	Hv0 = 1 / (1 / v0 + 1 / vw_F)
	Hv0 = max(Hv0, eps)
	Hv = 1 / (1 / v + 1 / vw_F)
	Hv = max(Hv, Hv0 + eps)

	Hv1 = 1 / L * (Hv - Hv0)
	Hv1 = max(Hv1, eps)

	DZ = Hv / HC_1z * (HD_1z + 1 / vw_F)
	FZ = Hv^(2) * (
			HF_1z * (HC_1z + HD_1z^(2) / HF_1z) / HC_1z +
			2 * HD_1z / (vw_F * HC_1z) +
			1 / (vw_F^(2) * HC_1z) +
			vw_T / vw_F^(2)
		)

	HZ = Hv1
	ChiZ = Hv0

	Res = (DZ = DZ, FZ = FZ, HZ = HZ, ChiZ = ChiZ)
	return Res
end