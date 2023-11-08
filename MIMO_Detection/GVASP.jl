module Module_GVASP

include("./Common.jl")

mutable struct Str_GVASP
	Hm_s_z::Vector{Float64}
	Hv_s_z1::Vector{Float64}
	Hv_s_z0::Vector{Float64}
	Hv_s_z::Vector{Float64}

	m_s_z::Vector{Float64}
	v_s_z0::Vector{Float64}
	v_s_z::Vector{Float64}

	Hm_p_z::Vector{Float64}
	Hv_p_z0::Vector{Float64}
	Hv_p_z::Vector{Float64}

	r_p_z_O::Vector{Float64}
	r_p_z::Vector{Float64}
	v_p_z0_inv_O::Vector{Float64}
	v_p_z0_inv::Vector{Float64}
	v_p_z_inv_O::Vector{Float64}
	v_p_z_inv::Vector{Float64}

	Hm_s_x::Vector{Float64}
	Hv_s_x0::Vector{Float64}
	Hv_s_x::Vector{Float64}

	m_s_x::Vector{Float64}
	v_s_x1::Vector{Float64}
	v_s_x0::Vector{Float64}
	v_s_x::Vector{Float64}

	Hm_p_x::Vector{Float64}
	Hv_p_x1::Vector{Float64}
	Hv_p_x0::Vector{Float64}
	Hv_p_x::Vector{Float64}

	r_p_x_O::Vector{Float64}
	r_p_x::Vector{Float64}
	v_p_x0_inv_O::Vector{Float64}
	v_p_x0_inv::Vector{Float64}
	v_p_x_inv_O::Vector{Float64}
	v_p_x_inv::Vector{Float64}

	Hm_p_tx::Vector{Float64}

	I_x0::Vector{Float64}

	HC_s_x0::Matrix{Float64}
	HC_s_x::Matrix{Float64}

	HC_p_tx0::Matrix{Float64}
	HC_p_tx::Matrix{Float64}

	HC_p_z0::Matrix{Float64}
	HC_p_z::Matrix{Float64}

	Mse_O::Float64

	H::Matrix{Float64}
	y::Vector{Float64}
end

function Setting(Prob)
	@extract Prob: N M H y

	return Str_GVASP(
			[zeros(M) for _ = 1 : 16] ...,
			[zeros(N) for _ = 1 : 19] ...,
			[zeros(N, N) for _ = 1 : 4] ...,
			[zeros(M, M) for _ = 1 : 2] ...,
			0, H, y
		)
end

function Init!(Gvasp::Str_GVASP, Prob, Prms)
	@extract Gvasp: r_p_z_O r_p_z v_p_z0_inv_O v_p_z0_inv v_p_z_inv_O v_p_z_inv
	@extract Gvasp: r_p_x_O r_p_x v_p_x0_inv_O v_p_x0_inv v_p_x_inv_O v_p_x_inv
	@extract Gvasp: I_x0
	@extract Gvasp: H y
	@extract Prms: L

	v1 = 1
	v0 = 1

	v = v0 + L * v1

	xIi = Init_X(H, y, Prob, Prms)
	zIi = H * xIi

	rx = xIi / v
	rz = zIi / v

	r_p_z_O .= rz
	r_p_z .= r_p_z_O
	v_p_z0_inv_O .= 1 / v0
	v_p_z0_inv .= v_p_z0_inv_O
	v_p_z_inv_O .= 1 / v
	v_p_z_inv .= v_p_z_inv_O

	r_p_x_O .= rx
	r_p_x .= r_p_x_O
	v_p_x0_inv_O .= 1 / v0
	v_p_x0_inv .= v_p_x0_inv_O
	v_p_x_inv_O .= 1 / v
	v_p_x_inv .= v_p_x_inv_O

	I_x0 .= xIi

	Gvasp.Mse_O = 1e3
end

function Iter!(Gvasp::Str_GVASP, t, Prms, io, x0)
	@extract Gvasp: Hm_s_z Hv_s_z1 Hv_s_z0 Hv_s_z m_s_z v_s_z0 v_s_z
	@extract Gvasp: Hm_p_z Hv_p_z0 Hv_p_z
	@extract Gvasp: r_p_z_O r_p_z v_p_z0_inv_O v_p_z0_inv v_p_z_inv_O v_p_z_inv
	@extract Gvasp: Hm_s_x Hv_s_x0 Hv_s_x m_s_x v_s_x1 v_s_x0 v_s_x
	@extract Gvasp: Hm_p_x Hv_p_x1 Hv_p_x0 Hv_p_x
	@extract Gvasp: r_p_x_O r_p_x v_p_x0_inv_O v_p_x0_inv v_p_x_inv_O v_p_x_inv
	@extract Gvasp: Hm_p_tx
	@extract Gvasp: HC_s_x0 HC_s_x
	@extract Gvasp: HC_p_tx0 HC_p_tx
	@extract Gvasp: HC_p_z0 HC_p_z
	@extract Gvasp: H y
	@extract Prms: L Mes Ax_F vw_F EPS Verb

	if Verb > 0
		println(io, "")
		println(io, "t: ", t)
	end

	Ht = H'

	m_p_z = r_p_z ./ v_p_z_inv
	v_p_z0 = 1 ./ v_p_z0_inv
	v_p_z = 1 ./ v_p_z_inv

	v_p_z1 = 1 / L * (v_p_z .- v_p_z0)

	if Verb > 0
		Pri_IO(io, "m_p_z: ", abs.(m_p_z))
		Pri_IO(io, "v_p_z0: ", v_p_z0)
		Pri_IO(io, "v_p_z: ", v_p_z)
		Pri_IO(io, "v_p_z1: ", v_p_z1)
	end

	@. v_p_z1 = max(v_p_z1, EPS)

	Out_Res = O_Est_1RSB(y, m_p_z, v_p_z1, v_p_z0, L, vw_F, EPS)
	Hm_s_z .= Out_Res.Hm
	Hv_s_z1 .= Out_Res.Hv1
	Hv_s_z0 .= Out_Res.Hv0

	if Verb > 0
		Pri_IO(io, "Hm_s_z: ", abs.(Hm_s_z))
		Pri_IO(io, "Hv_s_z1: ", Hv_s_z1)
		Pri_IO(io, "Hv_s_z0: ", Hv_s_z0)
	end

	@. Hv_s_z1 = max(Hv_s_z1, EPS)
	@. Hv_s_z0 = max(Hv_s_z0, EPS)

	Hv_s_z .= Hv_s_z0 .+ L * Hv_s_z1

	v_s_z0 .= Hv_s_z0 ./ (1 .- Hv_s_z0 .* v_p_z0_inv)
	v_s_z .= Hv_s_z ./ (1 .- Hv_s_z .* v_p_z_inv)

	if Verb > 0
		Pri_IO(io, "v_s_z0: ", v_s_z0)
		Pri_IO(io, "v_s_z: ", v_s_z)
	end

	@. v_s_z0 = max(v_s_z0, EPS)
	@. v_s_z = max(v_s_z, EPS)

	if Verb > 0
		tmp_v_s_z0 = v_s_z0 .+ EPS
		Res_v_s_z0 = sum(v_s_z .>= tmp_v_s_z0)
		println(io, "Res_v_s_z0: ", Res_v_s_z0)
	end

	@. v_s_z = max(v_s_z, v_s_z0 .+ EPS)

	if Verb > 0
		Pri_IO(io, "v_s_z: ", v_s_z)
	end

	m_s_z .= v_s_z .* (Hm_s_z ./ Hv_s_z .- r_p_z)

	if Verb > 0
		Pri_IO(io, "m_s_z: ", abs.(m_s_z))
	end

	HC_s_x0 .= inv(Ht * Diagonal(1 ./ v_s_z0) * H .+ Diagonal(v_p_x0_inv))
	HC_s_x .= inv(Ht * Diagonal(1 ./ v_s_z) * H .+ Diagonal(v_p_x_inv))
	Hm_s_x .= HC_s_x * (Ht * (m_s_z ./ v_s_z) .+ r_p_x)

	Hv_s_x0 .= diag(HC_s_x0)
	Hv_s_x .= diag(HC_s_x)

	if Verb > 0
		Pri_IO(io, "Hm_s_x: ", abs.(Hm_s_x))
		Pri_IO(io, "Hv_s_x0: ", Hv_s_x0)
		Pri_IO(io, "Hv_s_x: ", Hv_s_x)
	end

	@. Hv_s_x0 = max(Hv_s_x0, EPS)
	@. Hv_s_x = max(Hv_s_x, EPS)

	if Verb > 0
		tmp_Hv_s_x0 = Hv_s_x0 .+ EPS
		Res_Hv_s_x0 = sum(Hv_s_x .>= tmp_Hv_s_x0)
		println(io, "Res_Hv_s_x0: ", Res_Hv_s_x0)
	end

	@. Hv_s_x = max(Hv_s_x, Hv_s_x0 .+ EPS)

	# v_s_x0 .= Hv_s_x0 ./ (1 .- Hv_s_x0 .* v_p_x0_inv)
	# v_s_x .= Hv_s_x ./ (1 .- Hv_s_x .* v_p_x_inv)
	v_s_x0 .= Hv_s_x0 ./ max.(1 .- Hv_s_x0 .* v_p_x0_inv, EPS)
	v_s_x .= Hv_s_x ./ max.(1 .- Hv_s_x .* v_p_x_inv, EPS)

	if Verb > 0
		Pri_IO(io, "v_s_x0: ", v_s_x0)
		Pri_IO(io, "v_s_x: ", v_s_x)
	end

	@. v_s_x0 = max(v_s_x0, EPS)
	@. v_s_x = max(v_s_x, EPS)

	if Verb > 0
		tmp_v_s_x0 = v_s_x0 .+ EPS
		Res_v_s_x0 = sum(v_s_x .>= tmp_v_s_x0)
		println(io, "Res_v_s_x0: ", Res_v_s_x0)
	end

	@. v_s_x = max(v_s_x, v_s_x0 .+ EPS)
	
	v_s_x1 .= 1 / L * (v_s_x .- v_s_x0)

	if Verb > 0
		Pri_IO(io, "v_s_x1: ", v_s_x1)
	end

	@. v_s_x1 = max(v_s_x1, EPS)

	m_s_x .= v_s_x .* (Hm_s_x ./ Hv_s_x .- r_p_x)

	if Verb > 0
		Pri_IO(io, "m_s_x: ", abs.(m_s_x))
	end

	In_Res = I_Est_1RSB(m_s_x, v_s_x1, v_s_x0, L, Ax_F, EPS)
	Hm_p_x .= In_Res.Hm
	Hv_p_x1 .= In_Res.Hv1
	Hv_p_x0 .= In_Res.Hv0

	if Verb > 0
		Pri_IO(io, "Hm_p_x: ", abs.(Hm_p_x))
		Pri_IO(io, "Hv_p_x1: ", Hv_p_x1)
		Pri_IO(io, "Hv_p_x0: ", Hv_p_x0)
	end

	@. Hv_p_x1 = max(Hv_p_x1, EPS)
	@. Hv_p_x0 = max(Hv_p_x0, EPS)

	Hv_p_x .= Hv_p_x0 .+ L * Hv_p_x1

	Mse = norm(Hm_p_x .- x0)^(2) / norm(x0)^(2)
	Eps = abs(Mse - Gvasp.Mse_O)^(2) / Mse^(2)
	Rho = dot(Hm_p_x, x0) / sqrt(dot(Hm_p_x, Hm_p_x) * dot(x0, x0))

	A_D = mean(x0 .* Hm_p_x)
	A_F = mean(Hm_p_x.^(2))
	A_H = mean(Hv_p_x1)
	A_GsF = mean(Hv_p_x0)

	Gvasp.Mse_O = Mse

	if Verb > 0
		Pri_IO(io, "Hv_p_x: ", Hv_p_x)
		println(io, "Mse: ", Mse, ", Eps: ", Eps, ", Rho: ", Rho,
				", A_D: ", A_D, ", A_F: ", A_F, ", A_H: ", A_H, ", A_GsF: ", A_GsF
			)
	end

	println("GVASP - Iter: ", t, ", Mse: ", Mse, ", Eps: ", Eps, ", Rho: ", Rho,
			", A_D: ", A_D, ", A_F: ", A_F, ", A_H: ", A_H, ", A_GsF: ", A_GsF
		)

	v_p_x_inv .= (v_s_x .- Hv_p_x) ./ (v_s_x .* Hv_p_x)
	r_p_x .= (Hm_p_x .* v_s_x .- m_s_x .* Hv_p_x) ./ (v_s_x .* Hv_p_x)

	if Verb > 0
		Pri_IO(io, "v_p_x_inv: ", v_p_x_inv)
		Pri_IO(io, "r_p_x: ", abs.(r_p_x))
	end

	Idx_X = (v_p_x_inv .< EPS)
	v_p_x_inv .= Idx_X .* v_p_x_inv_O .+ (1 .- Idx_X) .* v_p_x_inv
	r_p_x .= Idx_X .* r_p_x_O .+ (1 .- Idx_X) .* r_p_x

	if Verb > 0
		println(io, "Idx_X: ", sum(Idx_X))
		Pri_IO(io, "v_p_x_inv: ", v_p_x_inv)
		Pri_IO(io, "r_p_x: ", abs.(r_p_x))
	end

	v_p_x_inv .= Mes * v_p_x_inv .+ (1 - Mes) * v_p_x_inv_O
	v_p_x_inv_O .= v_p_x_inv

	r_p_x .= Mes * r_p_x .+ (1 - Mes) * r_p_x_O
	r_p_x_O .= r_p_x

	v_p_x0_inv .= (v_s_x0 .- Hv_p_x0) ./ (v_s_x0 .* Hv_p_x0)

	if Verb > 0
		Pri_IO(io, "v_p_x0_inv: ", v_p_x0_inv)
	end

	Idx_x0 = (v_p_x0_inv .< EPS)
	v_p_x0_inv .= Idx_x0 .* v_p_x0_inv_O .+ (1 .- Idx_x0) .* v_p_x0_inv

	if Verb > 0
		println(io, "Idx_x0: ", sum(Idx_x0))
		Pri_IO(io, "v_p_x0_inv: ", v_p_x0_inv)
	end

	v_p_x0_inv .= Mes * v_p_x0_inv .+ (1 - Mes) * v_p_x0_inv_O
	v_p_x0_inv_O .= v_p_x0_inv

	T_m_p_x = r_p_x ./ v_p_x_inv
	T_v_p_x0 = 1 ./ v_p_x0_inv
	T_v_p_x = 1 ./ v_p_x_inv

	if Verb > 0
		tmp_T_v_p_x0 = T_v_p_x0 .+ EPS
		Res_T_v_p_x0 = sum(T_v_p_x .>= tmp_T_v_p_x0)
		println(io, "Res_T_v_p_x0: ", Res_T_v_p_x0)
	end

	@. T_v_p_x = max(T_v_p_x, T_v_p_x0 .+ EPS)

	v_p_x_inv .= 1 ./ T_v_p_x
	r_p_x .= T_m_p_x ./ T_v_p_x

	v_p_x_inv_O .= v_p_x_inv
	r_p_x_O .= r_p_x

	if Verb > 0
		Pri_IO(io, "v_p_x_inv: ", v_p_x_inv)
		Pri_IO(io, "r_p_x: ", abs.(r_p_x))
	end

	HC_p_tx0 .= inv(Ht * Diagonal(1 ./ v_s_z0) * H .+ Diagonal(v_p_x0_inv))
	HC_p_tx .= inv(Ht * Diagonal(1 ./ v_s_z) * H .+ Diagonal(v_p_x_inv))
	Hm_p_tx .= HC_p_tx * (Ht * (m_s_z ./ v_s_z) .+ r_p_x)

	if Verb > 0
		Pri_IO(io, "Hm_p_tx: ", abs.(Hm_p_tx))
	end

	HC_p_z0 .= H * HC_p_tx0 * Ht
	HC_p_z .= H * HC_p_tx * Ht
	Hm_p_z .= H * Hm_p_tx

	Hv_p_z0 .= diag(HC_p_z0)
	Hv_p_z .= diag(HC_p_z)

	if Verb > 0
		Pri_IO(io, "Hm_p_z: ", abs.(Hm_p_z))
		Pri_IO(io, "Hv_p_z0: ", Hv_p_z0)
		Pri_IO(io, "Hv_p_z: ", Hv_p_z)
	end

	@. Hv_p_z0 = max(Hv_p_z0, EPS)
	@. Hv_p_z = max(Hv_p_z, EPS)

	if Verb > 0
		tmp_Hv_p_z0 = Hv_p_z0 .+ EPS
		Res_Hv_p_z0 = sum(Hv_p_z .>= tmp_Hv_p_z0)
		println(io, "Res_Hv_p_z0: ", Res_Hv_p_z0)
	end

	@. Hv_p_z = max(Hv_p_z, Hv_p_z0 .+ EPS)

	v_p_z_inv .= (v_s_z .- Hv_p_z) ./ (v_s_z .* Hv_p_z)
	r_p_z .= (Hm_p_z .* v_s_z .- m_s_z .* Hv_p_z) ./ (v_s_z .* Hv_p_z)

	if Verb > 0
		Pri_IO(io, "v_p_z_inv: ", v_p_z_inv)
		Pri_IO(io, "r_p_z: ", abs.(r_p_z))
	end

	Idx_Z = (v_p_z_inv .< 0)
	v_p_z_inv .= Idx_Z .* v_p_z_inv_O .+ (1 .- Idx_Z) .* v_p_z_inv
	r_p_z .= Idx_Z .* r_p_z_O .+ (1 .- Idx_Z) .* r_p_z

	if Verb > 0
		println(io, "Idx_Z: ", sum(Idx_Z))
		Pri_IO(io, "v_p_z_inv: ", v_p_z_inv)
		Pri_IO(io, "r_p_z: ", abs.(r_p_z))
	end

	v_p_z_inv .= Mes * v_p_z_inv .+ (1 - Mes) * v_p_z_inv_O
	v_p_z_inv_O .= v_p_z_inv

	r_p_z .= Mes * r_p_z .+ (1 - Mes) * r_p_z_O
	r_p_z_O .= r_p_z

	v_p_z0_inv .= (v_s_z0 .- Hv_p_z0) ./ (v_s_z0 .* Hv_p_z0)

	if Verb > 0
		Pri_IO(io, "v_p_z0_inv: ", v_p_z0_inv)
	end

	Idx_z0 = (v_p_z0_inv .< 0)
	v_p_z0_inv .= Idx_z0 .* v_p_z0_inv_O .+ (1 .- Idx_z0) .* v_p_z0_inv

	if Verb > 0
		println(io, "Idx_z0: ", sum(Idx_z0))
		Pri_IO(io, "v_p_z0_inv: ", v_p_z0_inv)
	end

	v_p_z0_inv .= Mes * v_p_z0_inv .+ (1 - Mes) * v_p_z0_inv_O
	v_p_z0_inv_O .= v_p_z0_inv

	T_m_p_z = r_p_z ./ v_p_z_inv
	T_v_p_z0 = 1 ./ v_p_z0_inv
	T_v_p_z = 1 ./ v_p_z_inv

	if Verb > 0
		tmp_T_v_p_z0 = T_v_p_z0 .+ EPS
		Res_T_v_p_z0 = sum(T_v_p_z .>= tmp_T_v_p_z0)
		println(io, "Res_T_v_p_z0: ", Res_T_v_p_z0)
	end

	@. T_v_p_z = max(T_v_p_z, T_v_p_z0 .+ EPS)

	v_p_z_inv .= 1 ./ T_v_p_z
	r_p_z .= T_m_p_z ./ T_v_p_z

	v_p_z_inv_O .= v_p_z_inv
	r_p_z_O .= r_p_z

	if Verb > 0
		Pri_IO(io, "v_p_z_inv: ", v_p_z_inv)
		Pri_IO(io, "r_p_z: ", abs.(r_p_z))
	end

	return Mse, Eps, Rho, A_D, A_F, A_H, A_GsF
end

Solve(Prob; kws ...) = Algo(Prob, Str_1RSB_Params(; kws ...))

function Algo(Prob, Prms::Str_1RSB_Params)
	@extract Prob: N M H x0 y
	@extract Prms: Epochs Sd Epsilon Verb Name

	Sd > 0 && Random.seed!(Sd)

	### Print utilities ###
	Df = DataFrame(
			Epoch = Int[], Mse = Float64[], Eps = Float64[], Rho = Float64[],
			A_D = Float64[], A_F = Float64[], A_H = Float64[], A_GsF = Float64[]
		)

	Gvasp = Setting(Prob)
	Init!(Gvasp, Prob, Prms)

	Ok = false

	io = 0

	if Verb > 0
		rm("./$Name-S_GVASP.txt", force = true)
		io = open("./$Name-S_GVASP.txt", "w")

		rm("./$Name-S_Gvasp_Df.bson", force = true)
		rm("./$Name-S_Gvasp.bson", force = true)
		rm("./$Name-S_Gvasp_Prms.bson", force = true)
	end

	Mse_O = 1e3
	Eps_O = 1e3
	Rho_O = 1e3

	A_D_O = 1e3
	A_F_O = 1e3
	A_H_O = 1e3
	A_GsF_O = 1e3

	for i = 1 : Epochs
		Mse, Eps, Rho, A_D, A_F, A_H, A_GsF = Iter!(Gvasp, i, Prms, io, x0)

		# if (Mse > Mse_O)
		# 	break
		# end

		Report!(Df, i, Mse, Eps, Rho, A_D, A_F, A_H, A_GsF)

		Mse_O = Mse
		Eps_O = Eps
		Rho_O = Rho

		A_D_O = A_D
		A_F_O = A_F
		A_H_O = A_H
		A_GsF_O = A_GsF

		Ok = Eps < Epsilon

		if Ok
			break
		end
	end

	Num = size(Df, 1) + 1

	for i = Num : Epochs
		Report!(Df, i, Mse_O, Eps_O, Rho_O, A_D_O, A_F_O, A_H_O, A_GsF_O)
	end
	
	if Verb > 0
		S_Gvasp_Df = deepcopy(Df)
		S_Gvasp = deepcopy(Gvasp)
		S_Gvasp_Prms = deepcopy(Prms)

		BSON.@save "./$Name-S_Gvasp_Df.bson" S_Gvasp_Df
		BSON.@save "./$Name-S_Gvasp.bson" S_Gvasp
		BSON.@save "./$Name-S_Gvasp_Prms.bson" S_Gvasp_Prms

		close(io)
	end

	!Ok && @warn("Not converged!")
	return Df, Gvasp, Ok, Prms
end

end	# Module