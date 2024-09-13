module Module_VAMP

include("./Common.jl")

mutable struct Str_VAMP
	Hm_s_z::Vector{Float64}
	Hv_s_z::Vector{Float64}

	m_s_z::Vector{Float64}
	v_s_z::Vector{Float64}

	Hm_p_z::Vector{Float64}
	Hv_p_z::Vector{Float64}

	r_p_z_O::Vector{Float64}
	r_p_z::Vector{Float64}
	v_p_z_inv_O::Vector{Float64}
	v_p_z_inv::Vector{Float64}

	Hm_s_x::Vector{Float64}
	Hv_s_x::Vector{Float64}

	m_s_x::Vector{Float64}
	v_s_x::Vector{Float64}

	Hm_p_x::Vector{Float64}
	Hv_p_x::Vector{Float64}

	r_p_x_O::Vector{Float64}
	r_p_x::Vector{Float64}
	v_p_x_inv_O::Vector{Float64}
	v_p_x_inv::Vector{Float64}

	Hm_p_tx::Vector{Float64}

	I_x0::Vector{Float64}

	HC_s_x::Matrix{Float64}

	HC_p_tx::Matrix{Float64}

	HC_p_z::Matrix{Float64}

	Mse_O::Float64

	H::Matrix{Float64}
	y::Vector{Float64}
end

function Setting(Prob)
	@extract Prob: N M H y

	return Str_VAMP(
			[zeros(M) for _ = 1 : 10] ...,
			[zeros(N) for _ = 1 : 12] ...,
			[zeros(N, N) for _ = 1 : 2] ...,
			zeros(M, M),
			0, H, y
		)
end

function Init!(Vamp::Str_VAMP, Prob, Prms)
	@extract Vamp: r_p_z_O r_p_z v_p_z_inv_O v_p_z_inv
	@extract Vamp: r_p_x_O r_p_x v_p_x_inv_O v_p_x_inv
	@extract Vamp: I_x0
	@extract Vamp: H y

	v = 1

	xIi = Init_X(H, y, Prob, Prms)
	zIi = H * xIi

	rx = xIi / v
	rz = zIi / v

	r_p_z_O .= rz
	r_p_z .= r_p_z_O
	v_p_z_inv_O .= 1 / v
	v_p_z_inv .= v_p_z_inv_O

	r_p_x_O .= rx
	r_p_x .= r_p_x_O
	v_p_x_inv_O .= 1 / v
	v_p_x_inv .= v_p_x_inv_O

	I_x0 .= xIi

	Vamp.Mse_O = 1e3
end

function Iter!(Vamp::Str_VAMP, t, Prms, io, x0)
	@extract Vamp: Hm_s_z Hv_s_z m_s_z v_s_z
	@extract Vamp: Hm_p_z Hv_p_z
	@extract Vamp: r_p_z_O r_p_z v_p_z_inv_O v_p_z_inv
	@extract Vamp: Hm_s_x Hv_s_x m_s_x v_s_x
	@extract Vamp: Hm_p_x Hv_p_x
	@extract Vamp: r_p_x_O r_p_x v_p_x_inv_O v_p_x_inv
	@extract Vamp: Hm_p_tx
	@extract Vamp: HC_s_x
	@extract Vamp: HC_p_tx
	@extract Vamp: HC_p_z
	@extract Vamp: H y
	@extract Prms: Mes Ax_T Cx_T vw_T eps Verb

	if Verb > 0
		println(io, "")
		println(io, "t: ", t)
	end

	Ht = H'

	m_p_z = r_p_z ./ v_p_z_inv
	v_p_z = 1 ./ v_p_z_inv

	if Verb > 0
		Pri_IO(io, "m_p_z: ", abs.(m_p_z))
		Pri_IO(io, "v_p_z: ", v_p_z)
	end

	Out_Res = O_Est_RS(y, m_p_z, v_p_z, vw_T, eps)
	Hm_s_z .= Out_Res.Hm
	Hv_s_z .= Out_Res.Hv

	if Verb > 0
		Pri_IO(io, "Hm_s_z: ", abs.(Hm_s_z))
		Pri_IO(io, "Hv_s_z: ", Hv_s_z)
	end

	@. Hv_s_z = max(Hv_s_z, eps)

	if Verb > 0
		Pri_IO(io, "Hv_s_z: ", Hv_s_z)
	end

	v_s_z .= Hv_s_z ./ (1 .- Hv_s_z .* v_p_z_inv)

	if Verb > 0
		Pri_IO(io, "v_s_z: ", v_s_z)
	end

	@. v_s_z = max(v_s_z, eps)

	m_s_z .= v_s_z .* (Hm_s_z ./ Hv_s_z .- r_p_z)

	if Verb > 0
		Pri_IO(io, "v_s_z: ", v_s_z)
		Pri_IO(io, "m_s_z: ", abs.(m_s_z))
	end

	HC_s_x .= inv(Ht * Diagonal(1 ./ v_s_z) * H .+ Diagonal(v_p_x_inv))
	Hm_s_x .= HC_s_x * (Ht * (m_s_z ./ v_s_z) .+ r_p_x)

	Hv_s_x .= diag(HC_s_x)

	if Verb > 0
		Pri_IO(io, "Hm_s_x: ", abs.(Hm_s_x))
		Pri_IO(io, "Hv_s_x: ", Hv_s_x)
	end

	@. Hv_s_x = max(Hv_s_x, eps)

	if Verb > 0
		Pri_IO(io, "Hv_s_x: ", Hv_s_x)
	end

	# v_s_x .= Hv_s_x ./ (1 .- Hv_s_x .* v_p_x_inv)
	v_s_x .= Hv_s_x ./ max.(1 .- Hv_s_x .* v_p_x_inv, eps)

	if Verb > 0
		Pri_IO(io, "v_s_x: ", v_s_x)
	end

	@. v_s_x = max(v_s_x, eps)

	m_s_x .= v_s_x .* (Hm_s_x ./ Hv_s_x .- r_p_x)

	if Verb > 0
		Pri_IO(io, "v_s_x: ", v_s_x)
		Pri_IO(io, "m_s_x: ", abs.(m_s_x))
	end

	In_Res = I_Est_RS(m_s_x, v_s_x, Ax_T, Cx_T, eps)
	Hm_p_x .= In_Res.Hm
	Hv_p_x .= In_Res.Hv

	if Verb > 0
		Pri_IO(io, "Hm_p_x: ", abs.(Hm_p_x))
		Pri_IO(io, "Hv_p_x: ", Hv_p_x)
	end

	@. Hv_p_x = max(Hv_p_x, eps)

	if Verb > 0
		Pri_IO(io, "Hv_p_x: ", Hv_p_x)
	end

	Mse = norm(Hm_p_x .- x0)^(2) / norm(x0)^(2)
	Eps = abs(Mse - Vamp.Mse_O)^(2) / Mse^(2)
	Rho = dot(Hm_p_x, x0) / sqrt(dot(Hm_p_x, Hm_p_x) * dot(x0, x0))

	A_D = mean(x0 .* Hm_p_x)
	A_F = mean(Hm_p_x.^(2))
	A_H = 0
	A_Chi = mean(Hv_p_x)

	Vamp.Mse_O = Mse

	if Verb > 0
		Pri_IO(io, "Hv_p_x: ", Hv_p_x)
		println(io, "Mse: ", Mse, ", Eps: ", Eps, ", Rho: ", Rho,
				", A_D: ", A_D, ", A_F: ", A_F, ", A_H: ", A_H, ", A_Chi: ", A_Chi
			)
	end

	println("VAMP - Iter: ", t, ", Mse: ", Mse, ", Eps: ", Eps, ", Rho: ", Rho,
			", A_D: ", A_D, ", A_F: ", A_F, ", A_H: ", A_H, ", A_Chi: ", A_Chi
		)

	v_p_x_inv .= (v_s_x .- Hv_p_x) ./ (v_s_x .* Hv_p_x)
	r_p_x .= (Hm_p_x .* v_s_x .- m_s_x .* Hv_p_x) ./ (v_s_x .* Hv_p_x)

	if Verb > 0
		Pri_IO(io, "v_p_x_inv: ", v_p_x_inv)
		Pri_IO(io, "r_p_x: ", abs.(r_p_x))
	end

	Idx_X = (v_p_x_inv .< eps)
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

	HC_p_tx .= inv(Ht * Diagonal(1 ./ v_s_z) * H .+ Diagonal(v_p_x_inv))
	Hm_p_tx .= HC_p_tx * (Ht * (m_s_z ./ v_s_z) .+ r_p_x)

	if Verb > 0
		Pri_IO(io, "Hm_p_tx: ", abs.(Hm_p_tx))
	end

	HC_p_z .= H * HC_p_tx * Ht
	Hm_p_z .= H * Hm_p_tx

	Hv_p_z .= diag(HC_p_z)

	if Verb > 0
		Pri_IO(io, "Hm_p_z: ", abs.(Hm_p_z))
		Pri_IO(io, "Hv_p_z: ", Hv_p_z)
	end

	@. Hv_p_z = max(Hv_p_z, eps)

	if Verb > 0
		Pri_IO(io, "Hv_p_z: ", Hv_p_z)
	end

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

	return Mse, Eps, Rho, A_D, A_F, A_H, A_Chi
end

Solve(Prob; kws ...) = Algo(Prob, Str_RS_Params(; kws ...))

function Algo(Prob, Prms::Str_RS_Params)
	@extract Prob: N M H x0 y
	@extract Prms: Epochs Sd Epsilon Verb Name

	Sd > 0 && Random.seed!(Sd)

	### Print utilities ###
	Df = DataFrame(
			Epoch = Int[], Mse = Float64[], Eps = Float64[], Rho = Float64[],
			A_D = Float64[], A_F = Float64[], A_H = Float64[], A_Chi = Float64[]
		)

	Vamp = Setting(Prob)
	Init!(Vamp, Prob, Prms)

	Ok = false

	io = 0

	if Verb > 0
		rm("./$Name-S_VAMP.txt", force = true)
		io = open("./$Name-S_VAMP.txt", "w")

		rm("./$Name-S_Vamp_Df.bson", force = true)
		rm("./$Name-S_Vamp.bson", force = true)
		rm("./$Name-S_Vamp_Prms.bson", force = true)
	end

	Mse_O = 1e3
	Eps_O = 1e3
	Rho_O = 1e3

	A_D_O = 1e3
	A_F_O = 1e3
	A_H_O = 1e3
	A_Chi_O = 1e3

	for i = 1 : Epochs
		Mse, Eps, Rho, A_D, A_F, A_H, A_Chi = Iter!(Vamp, i, Prms, io, x0)

		Report!(Df, i, Mse, Eps, Rho, A_D, A_F, A_H, A_Chi)

		Mse_O = Mse
		Eps_O = Eps
		Rho_O = Rho

		A_D_O = A_D
		A_F_O = A_F
		A_H_O = A_H
		A_Chi_O = A_Chi

		Ok = Eps < Epsilon

		if Ok
			break
		end
	end

	Num = size(Df, 1) + 1

	for i = Num : Epochs
		Report!(Df, i, Mse_O, Eps_O, Rho_O, A_D_O, A_F_O, A_H_O, A_Chi_O)
	end
	
	if Verb > 0
		S_Vamp_Df = deepcopy(Df)
		S_Vamp = deepcopy(Vamp)
		S_Vamp_Prms = deepcopy(Prms)

		BSON.@save "./$Name-S_Vamp_Df.bson" S_Vamp_Df
		BSON.@save "./$Name-S_Vamp.bson" S_Vamp
		BSON.@save "./$Name-S_Vamp_Prms.bson" S_Vamp_Prms

		close(io)
	end

	!Ok && @warn("Not converged!")
	return Df, Vamp, Ok, Prms
end

end	# Module