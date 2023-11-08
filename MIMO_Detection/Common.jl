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

function Df_Push(Epochs, Mse_List, Eps_List, Rho_List, A_D_List, A_F_List, A_H_List, A_GsF_List)
	Mse_Iter = mean(Mse_List, dims = 2)
	Eps_Iter = mean(Eps_List, dims = 2)
	Rho_Iter = mean(Rho_List, dims = 2)
	A_D_Iter = mean(A_D_List, dims = 2)
	A_F_Iter = mean(A_F_List, dims = 2)
	A_H_Iter = mean(A_H_List, dims = 2)
	A_GsF_Iter = mean(A_GsF_List, dims = 2)

	Df = DataFrame(
			Epoch = Int[], Mse = Float64[], Eps = Float64[], Rho = Float64[],
			A_D = Float64[], A_F = Float64[], A_H = Float64[], A_GsF = Float64[]
		)

	for i = 1 : Epochs
		Res = (
				Epoch = i, Mse = Mse_Iter[i, 1], Eps = Eps_Iter[i, 1], Rho = Rho_Iter[i, 1],
				A_D = A_D_Iter[i, 1], A_F = A_F_Iter[i, 1], A_H = A_H_Iter[i, 1], A_GsF = A_GsF_Iter[i, 1],
			)
		push!(Df, Res)
	end

	return Df
end

function Df_Push_New(Epochs, Mse_List, Eps_List, Rho_List,
	A_D_List, A_F_List, A_H_List, A_GsF_List,
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
	A_GsF_Iter = mean(A_GsF_List, dims = 2)

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
			A_D = Float64[], A_F = Float64[], A_H = Float64[], A_GsF = Float64[],
			S_v_s_z = Float64[], S_HD_2z = Float64[], S_HF_2z = Float64[],
			S_v_s_x = Float64[], S_HD_1x = Float64[], S_HF_1x = Float64[],
			S_v_p_x = Float64[], S_HD_2x = Float64[], S_HF_2x = Float64[],
			S_v_p_z = Float64[], S_HD_1z = Float64[], S_HF_1z = Float64[]
		)

	for i = 1 : Epochs
		Res = (
				Epoch = i, Mse = Mse_Iter[i, 1], Eps = Eps_Iter[i, 1], Rho = Rho_Iter[i, 1],
				A_D = A_D_Iter[i, 1], A_F = A_F_Iter[i, 1], A_H = A_H_Iter[i, 1], A_GsF = A_GsF_Iter[i, 1],
				S_v_s_z = S_v_s_z_Iter[i, 1], S_HD_2z = S_HD_2z_Iter[i, 1], S_HF_2z = S_HF_2z_Iter[i, 1],			
				S_v_s_x = S_v_s_x_Iter[i, 1], S_HD_1x = S_HD_1x_Iter[i, 1], S_HF_1x = S_HF_1x_Iter[i, 1],			
				S_v_p_x = S_v_p_x_Iter[i, 1], S_HD_2x = S_HD_2x_Iter[i, 1], S_HF_2x = S_HF_2x_Iter[i, 1],			
				S_v_p_z = S_v_p_z_Iter[i, 1], S_HD_1z = S_HD_1z_Iter[i, 1], S_HF_1z = S_HF_1z_Iter[i, 1],			
			)
		push!(Df, Res)
	end

	return Df
end

function Report!(Df, Epoch, Mse, Eps, Rho, A_D, A_F, A_H, A_GsF)
	Res = (
			Epoch = Epoch, Mse = Mse, Eps = Eps, Rho = Rho,
			A_D = A_D, A_F = A_F, A_H = A_H, A_GsF = A_GsF
		)
	push!(Df, Res)
end

function Report_New!(Df, Epoch, Mse, Eps, Rho, A_D, A_F, A_H, A_GsF,
	S_v_s_z, S_HD_2z, S_HF_2z,
	S_v_s_x, S_HD_1x, S_HF_1x,
	S_v_p_x, S_HD_2x, S_HF_2x,
	S_v_p_z, S_HD_1z, S_HF_1z)
	Res = (
			Epoch = Epoch, Mse = Mse, Eps = Eps, Rho = Rho,
			A_D = A_D, A_F = A_F, A_H = A_H, A_GsF = A_GsF,
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
	vw_T::Float64

	cor_flag::Bool
	cor::Float64

	H::Matrix		# measured vector of size M * N
	x0::Vector		# measured vector of size N * 1
	z::Vector		# measured vector of size M * 1
	w::Vector		# measured vector of size M * 1
	y::Vector		# measured vector of size M * 1
end

Problem_Setting(name::String; kws ...) = name == "GLE" ? GLE(; kws ...) : error("Uknown problem")

### Generilized linear estimation ###
function GLE(; Prior = :BPSK,
		N = 0, a = 0, Seed = 0,
		Ax_T = 0,
		vw_T = 0,
		cor_flag = 0,
		cor = 0
	)

	Seed > 0 && Random.seed!(Seed)

	M = round(Int, N * a)

	if Prior == :BPSK
		x0 = rand((- Ax_T, Ax_T), N)
	end

	H = sqrt(1 / N) * randn(M, N)

	if cor_flag
		Rx = Get_Rx(N, cor)
		H = H * Rx
	end

	z = H * x0
	w = sqrt(vw_T) * randn(M)
	y = z .+ w

	return Str_Problem(
			Prior,
			a, N, M,
			Seed,
			Ax_T, vw_T,
			cor_flag, cor,
			H, x0, z, w, y
		)
end

# Based on Li Ping and Cheng
function I_Est_RS(m, v, Ax_F, EPS)
	N = size(m, 1)

	Hm = Ax_F * sign.(m)

	List = zeros(N, 2)
	List[:, 1] .= abs.(m .+ Ax_F).^(2)
	List[:, 2] .= abs.(m .- Ax_F).^(2)

	v_phi = mean(List)

	xi = sqrt(v_phi) * randn(N)
	x = rand((- Ax_F, Ax_F), N)

	Hm_N = Ax_F * sign.(x .+ xi)

	xi_Hm = xi' * Hm_N

	div = abs(xi_Hm) / (N * v_phi)

	Hv = div * v

	Res = (Hm = Hm, Hv = Hv)
	return Res
end

function I_Est_RS1(m, v, Ax_F, EPS)
	Hm = Ax_F * sign.(m)
	Hv = EPS

	Res = (Hm = Hm, Hv = Hv)
	return Res
end

function Log_Phi(m, v1, v0, L, Ax_F)
	Zs = exp(
			- L * (m - Ax_F)^(2) / (
				2 * (v0 + L * v1)
			)
		) * H_Func(
			- (
				Ax_F * L * v1 + m * v0
			) / sqrt(
				v0 * v1 * (v0 + L * v1)
			)
		)

	Zp = exp(
			- L * (m + Ax_F)^(2) / (
				2 * (v0 + L * v1)
			)
		) * H_Func(
			- (
				Ax_F * L * v1 - m * v0
			) / sqrt(
				v0 * v1 * (v0 + L * v1)
			)
		)

	lp = - 1 / 2 * log(v0) -
		log(
			H_Func(- Ax_F / sqrt(v0))
		) -
		1 / (2 * L) * log(v0 + L * v1) +
		1 / (2 * L) * log(v0) +
		1 / L * log(Zs + Zp)

	return lp
end

### Use the derivative from replica computations ###
P_m(m, v1, v0, L, Ax_F) = ForwardDiff.derivative(m -> Log_Phi(m, v1, v0, L, Ax_F), m)
P_v1(m, v1, v0, L, Ax_F) = ForwardDiff.derivative(v1 -> Log_Phi(m, v1, v0, L, Ax_F), v1)
P_v0(m, v1, v0, L, Ax_F) = ForwardDiff.derivative(v0 -> Log_Phi(m, v1, v0, L, Ax_F), v0)

function I_Est_1RSB(m, v1, v0, L, Ax_F, EPS)
	v = v0 .+ L * v1

	pm = @. P_m(m, v1, v0, L, Ax_F)
	pv1 = @. P_v1(m, v1, v0, L, Ax_F)
	pv0 = @. P_v0(m, v1, v0, L, Ax_F)
	# pm2 = - (- 2 * pv1 .+ L * pm.^(2))

	Tau0 = 1 / (L - 1) * (2 * pv1 .- 2 * L * pv0)
	# Tau = - pm2
	Tau = - 2 * pv1 .+ L * abs.(pm).^(2)

	Hm = pm .* v .+ m

	Hv0 = v0 .- Tau0 .* abs.(v0).^(2)
	@. Hv0 = max(Hv0, EPS)
	Hv = v .- Tau .* abs.(v).^(2)
	@. Hv = max(Hv, Hv0 + EPS)

	Hv1 = 1 / L * (Hv .- Hv0)
	@. Hv1 = max(Hv1, EPS)

	Res = (Hm = Hm, Hv1 = Hv1, Hv0 = Hv0)

	return Res
end

function O_Est_RS(y, m, v, vw_F, EPS)
	Hv = 1 ./ (1 ./ v .+ 1 / vw_F)
	@. Hv = max(Hv, EPS)

	Hm = Hv .* (m ./ v .+ y / vw_F)

	Res = (Hm = Hm, Hv = Hv)
	return Res
end

function O_Est_1RSB(y, m, v1, v0, L, vw_F, EPS)
	v = v0 .+ L * v1

	Hv0 = 1 ./ (1 ./ v0 .+ 1 / vw_F)
	@. Hv0 = max(Hv0, EPS)

	Hv = 1 ./ (1 ./ v .+ 1 / vw_F)
	@. Hv = max(Hv, Hv0 .+ EPS)

	Hv1 = 1 / L * (Hv .- Hv0)
	@. Hv1 = max(Hv1, EPS)

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
	Ax_F = 0
	vw_F = 0			# the variance of likelihood function
	EPS = 0
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
	vw_F = 0			# the variance of likelihood function
	EPS = 0
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

function First_Integral_Base(Func, Min_X, Max_X)
	Res = quadgk(
			w -> begin
				tmp = Gauss(w, 0, 1) * Func(w)
				isfinite(tmp) ? tmp : 0.
			end,
			Min_X, Max_X, rtol = 1e-8, atol = 1e-9, maxevals = 0
		)

	return Res[1]
end

function First_Integral(Func)
	dx = 1e-2
	# dist = 40
	dist = 10
	List = map(
			x -> sign(x) * abs(x)^(2), -1 : dx : 1
		) * dist

	Ints = [
			(List[i], List[i + 1]) for i = 1 : length(List) - 1
		]

	Res = sum(
		Ip -> begin
			Min_X = Ip[1]
			Max_X = Ip[2]
			First_Integral_Base(Func, Min_X, Max_X)
		end, Ints
	)

	return Res
end

function SE_Hm(m, v1, v0, L, Ax_F)
	v = v0 + L * v1

	pm = P_m(m, v1, v0, L, Ax_F)
	Hm = pm * v + m

	return Hm
end

function SE_Hv1(m, v1, v0, L, Ax_F)
	v = v0 + L * v1

	pm = P_m(m, v1, v0, L, Ax_F)
	pv1 = P_v1(m, v1, v0, L, Ax_F)
	pv0 = P_v0(m, v1, v0, L, Ax_F)

	Tau0 = 1 / (L - 1) * (2 * pv1 - 2 * L * pv0)
	Tau = - 2 * pv1 + L * pm^(2)

	Hv0 = v0 - Tau0 * v0^(2)
	Hv = v - Tau * v^(2)

	Hv1 = 1 / L * (Hv - Hv0)

	return Hv1
end

function SE_Hv0(m, v1, v0, L, Ax_F)

	pv1 = P_v1(m, v1, v0, L, Ax_F)
	pv0 = P_v0(m, v1, v0, L, Ax_F)

	Tau0 = 1 / (L - 1) * (2 * pv1 - 2 * L * pv0)

	Hv0 = v0 - Tau0 * v0^(2)

	return Hv0
end

function I_SE_1RSB(HD_1x, HF_1x, v1, v0, L, Ax_T, Ax_F, EPS)
	v = v0 + L * v1
	r = 1 / v

	DX = 1 / 2 * Ax_T * First_Integral(
			w -> begin
				SE_Hm(- HD_1x / r * Ax_T + sqrt(- HF_1x / r^(2)) * w, v1, v0, L, Ax_F)
			end
		) - 1 / 2 * Ax_T * First_Integral(
			w -> begin
				SE_Hm(HD_1x / r * Ax_T + sqrt(- HF_1x / r^(2)) * w, v1, v0, L, Ax_F)
			end
		)

	FX = 1 / 2 * First_Integral(
			w -> begin
				(
					SE_Hm(- HD_1x / r * Ax_T + sqrt(- HF_1x / r^(2)) * w, v1, v0, L, Ax_F)
				)^(2)
			end
		) + 1 / 2 * First_Integral(
			w -> begin
				(
					SE_Hm(HD_1x / r * Ax_T + sqrt(- HF_1x / r^(2)) * w, v1, v0, L, Ax_F)
				)^(2)
			end
		)

	HX = 1 / 2 * First_Integral(
			w -> begin
				SE_Hv1(- HD_1x / r * Ax_T + sqrt(- HF_1x / r^(2)) * w, v1, v0, L, Ax_F)
			end
		) + 1 / 2 * First_Integral(
			w -> begin
				SE_Hv1(HD_1x / r * Ax_T + sqrt(- HF_1x / r^(2)) * w, v1, v0, L, Ax_F)
			end
		)

	GXsFX = 1 / 2 * First_Integral(
			w -> begin
				SE_Hv0(- HD_1x / r * Ax_T + sqrt(- HF_1x / r^(2)) * w, v1, v0, L, Ax_F)
			end
		) + 1 / 2 * First_Integral(
			w -> begin
				SE_Hv0(HD_1x / r * Ax_T + sqrt(- HF_1x / r^(2)) * w, v1, v0, L, Ax_F)
			end
		)

	Res = (DX = DX, FX = FX, HX = HX, GXsFX = GXsFX)
	return Res
end

function O_SE_1RSB(HC_1z, HD_1z, HF_1z, v1, v0, L, vT, vF, EPS)
	v = v0 + L * v1

	Hv0 = 1 / (1 / v0 + 1 / vF)
	Hv0 = max(Hv0, EPS)
	Hv = 1 / (1 / v + 1 / vF)
	Hv = max(Hv, Hv0 + EPS)

	Hv1 = 1 / L * (Hv - Hv0)
	Hv1 = max(Hv1, EPS)

	DZ = Hv / HC_1z * (1 / vF - HD_1z)
	FZ = Hv^(2) * (
			- HF_1z * (HC_1z - HD_1z^(2) / HF_1z) / HC_1z +
			1 / (vF^(2) * HC_1z) +
			vT / vF^(2) -
			2 * HD_1z / (vF * HC_1z)
		)

	HZ = Hv1
	GZsFZ = Hv0

	Res = (DZ = DZ, FZ = FZ, HZ = HZ, GZsFZ = GZsFZ)
	return Res
end