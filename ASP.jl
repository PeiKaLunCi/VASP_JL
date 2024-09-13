module Module_ASP

include("./Common.jl")

mutable struct Str_ASP
	Z_a_O::Vector{Float64}
	Z_a::Vector{Float64}
	V_a1::Vector{Float64}
	V_a0_O::Vector{Float64}
	V_a0::Vector{Float64}
	V_a_O::Vector{Float64}
	V_a::Vector{Float64}

	Tz_a::Vector{Float64}
	Tv_a1::Vector{Float64}
	Tv_a0::Vector{Float64}
	Tv_a::Vector{Float64}

	s_a::Vector{Float64}
	t_a0::Vector{Float64}
	t_a::Vector{Float64}

	Tm_i::Vector{Float64}
	Tv_i1::Vector{Float64}
	Tv_i0::Vector{Float64}
	Tv_i::Vector{Float64}

	Hm_i::Vector{Float64}
	Hv_i1::Vector{Float64}
	Hv_i0::Vector{Float64}
	Hv_i::Vector{Float64}

	I_x0::Vector{Float64}

	Mse_O::Float64
	Eps_O::Float64
	Rho_O::Float64
	A_D_O::Float64
	A_F_O::Float64
	A_H_O::Float64
	A_Chi_O::Float64

	H::Matrix{Float64}
	y::Vector{Float64}
end

function Setting(Prob)
	@extract Prob: N M H y

	return Str_ASP(
			[zeros(M) for _ = 1 : 14] ...,
			[zeros(N) for _ = 1 : 9] ...,
			0, 0, 0, 0, 0, 0, 0, H, y
		)
end

function Init!(Asp::Str_ASP, Prob, Prms)
	@extract Asp: Z_a_O V_a0_O V_a_O s_a
	@extract Asp: Hm_i Hv_i1 Hv_i0 Hv_i I_x0
	@extract Asp: H y
	@extract Prms: L

	v1 = 1
	v0 = 1

	Z_a_O .= 0
	V_a0_O .= v0
	V_a_O .= v0 + L * v1

	s_a .= 0

	xIi = Init_X(H, y, Prob, Prms)

	Hm_i .= xIi

	Hv_i1 .= v1
	Hv_i0 .= v0
	Hv_i .= v0 + L * v1

	I_x0 .= xIi

	Asp.Mse_O = 1e3
	Asp.Eps_O = 1e3
	Asp.Rho_O = 1e3
	Asp.A_D_O = 1e3
	Asp.A_F_O = 1e3
	Asp.A_H_O = 1e3
	Asp.A_Chi_O = 1e3
end

function Iter!(Asp::Str_ASP, t, Prms, io, x0)
	@extract Asp: Z_a_O Z_a V_a1 V_a0_O V_a0 V_a_O V_a
	@extract Asp: Tz_a Tv_a1 Tv_a0 Tv_a s_a t_a0 t_a
	@extract Asp: Tm_i Tv_i1 Tv_i0 Tv_i Hm_i Hv_i1 Hv_i0 Hv_i
	@extract Asp: H y
	@extract Prms: L Mes Ax_F Cx_F vw_F eps Verb

	if Verb > 0
		println(io, "")
		println(io, "t: ", t)
	end

	Ht = H'
	H2 = abs.(H).^(2)
	H2t = H2'

	V_a0 .= H2 * Hv_i0
	V_a .= H2 * Hv_i

	if Verb > 0
		Pri_IO(io, "V_a0: ", V_a0)
		Pri_IO(io, "V_a: ", V_a)

		tmp_V_a1 = 1 / L * (V_a .- V_a0)
		Pri_IO(io, "tmp_V_a1: ", tmp_V_a1)
	end

	Z_a .= H * Hm_i .- s_a .* V_a

	if Verb > 0
		Pri_IO(io, "Z_a: ", abs.(Z_a))
	end

	V_a0 .= Mes * V_a0 .+ (1 - Mes) * V_a0_O
	V_a0_O .= V_a0
	V_a .= Mes * V_a .+ (1 - Mes) * V_a_O
	V_a_O .= V_a

	Z_a .= Mes * Z_a .+ (1 - Mes) * Z_a_O
	Z_a_O .= Z_a

	if Verb > 0
		tmp_V_a0 = V_a0 .+ eps
		Res_V_a0 = sum(V_a .>= tmp_V_a0)
		println(io, "Res_V_a0: ", Res_V_a0)
	end

	@. V_a = max(V_a, V_a0 + eps)
	V_a_O .= V_a

	V_a1 .= 1 / L * (V_a .- V_a0)

	if Verb > 0
		Pri_IO(io, "V_a1: ", V_a1)
	end

	@. V_a1 = max(V_a1, eps)

	Out_Res = O_Est_1RSB(y, Z_a, V_a1, V_a0, L, vw_F, eps)
	Tz_a .= Out_Res.Hm
	Tv_a1 .= Out_Res.Hv1
	Tv_a0 .= Out_Res.Hv0

	a = isinf.(Tz_a) .+ isnan.(Tz_a) .+
		isinf.(Tv_a1) .+ isnan.(Tv_a1) .+
		isinf.(Tv_a0) .+ isnan.(Tv_a0)
	Idx_a = sum(a)

	if Idx_a > 0
		if Verb > 0
			println(io, "Z-break: ", Idx_a)
		end
		return 1e3, Asp.Eps_O, Asp.Rho_O, Asp.A_D_O, Asp.A_F_O, Asp.A_H_O, Asp.A_Chi_O
	end

	if Verb > 0
		Pri_IO(io, "Tz_a: ", abs.(Tz_a))
		Pri_IO(io, "Tv_a1: ", Tv_a1)
		Pri_IO(io, "Tv_a0: ", Tv_a0)
	end

	@. Tv_a1 = max(Tv_a1, eps)
	@. Tv_a0 = max(Tv_a0, eps)

	if Verb > 0
		Pri_IO(io, "Tv_a1: ", Tv_a1)
		Pri_IO(io, "Tv_a0: ", Tv_a0)
	end

	Tv_a .= Tv_a0 .+ L * Tv_a1

	s_a .= (Tz_a .- Z_a) ./ V_a
	t_a0 .= (V_a0 .- Tv_a0) ./ abs.(V_a0).^(2)
	t_a .= (V_a .- Tv_a) ./ abs.(V_a).^(2)

	if Verb > 0
		tmp_t_a1 = 1 / L * (t_a0 .- t_a)

		Pri_IO(io, "Tv_a: ", Tv_a)
		Pri_IO(io, "s_a: ", abs.(s_a))
		Pri_IO(io, "t_a0: ", t_a0)
		Pri_IO(io, "t_a: ", t_a)
		Pri_IO(io, "tmp_t_a1: ", tmp_t_a1)
	end

	@. t_a0 = max(t_a0, eps)
	@. t_a = max(t_a, eps)

	if Verb > 0
		Pri_IO(io, "t_a0: ", t_a0)
		Pri_IO(io, "t_a: ", t_a)
	end

	@. t_a = min(t_a, t_a0)

	if Verb > 0
		Pri_IO(io, "t_a: ", t_a)
	end

	Tv_i0 .= 1 ./ (H2t * t_a0)
	Tv_i .= 1 ./ (H2t * t_a)

	if Verb > 0
		tmp_A0 = H2t * t_a0
		tmp_A = H2t * t_a

		tmp_A1 = 1 / L * (tmp_A0 .- tmp_A)

		tmp_Tv_i1 = 1 / L * (Tv_i .- Tv_i0)

		Pri_IO(io, "Tv_i0: ", Tv_i0)
		Pri_IO(io, "Tv_i: ", Tv_i)
		Pri_IO(io, "tmp_A0: ", tmp_A0)
		Pri_IO(io, "tmp_A: ", tmp_A)
		Pri_IO(io, "tmp_A1: ", tmp_A1)
		Pri_IO(io, "tmp_Tv_i1: ", tmp_Tv_i1)
	end

	@. Tv_i0 = max(Tv_i0, eps)
	@. Tv_i = max(Tv_i, eps)

	if Verb > 0
		Pri_IO(io, "Tv_i0: ", Tv_i0)
		Pri_IO(io, "Tv_i: ", Tv_i)
	end

	@. Tv_i = max(Tv_i, Tv_i0 + eps)

	if Verb > 0
		Pri_IO(io, "Tv_i: ", Tv_i)
	end

	Tv_i1 .= 1 / L * (Tv_i .- Tv_i0)

	if Verb > 0
		Pri_IO(io, "Tv_i1: ", Tv_i1)
	end

	@. Tv_i1 = max(Tv_i1, eps)

	if Verb > 0
		Pri_IO(io, "Tv_i1: ", Tv_i1)
	end

	Tm_i .= Hm_i .+ Tv_i .* (Ht * s_a)

	In_Res = I_Est_1RSB(Tm_i, Tv_i1, Tv_i0, L, Ax_F, Cx_F, eps)
	Hm_i .= In_Res.Hm
	Hv_i1 .= In_Res.Hv1
	Hv_i0 .= In_Res.Hv0

	a = isinf.(Hm_i) .+ isnan.(Hm_i) .+
		isinf.(Hv_i1) .+ isnan.(Hv_i1) .+
		isinf.(Hv_i0) .+ isnan.(Hv_i0)
	Idx_a = sum(a)

	if Idx_a > 0
		if Verb > 0
			println(io, "X-break: ", Idx_a)
		end
		return 1e3, Asp.Eps_O, Asp.Rho_O, Asp.A_D_O, Asp.A_F_O, Asp.A_H_O, Asp.A_Chi_O
	end

	if Verb > 0
		tmp_B = Tm_i ./ Tv_i

		Pri_IO(io, "Tm_i: ", abs.(Tm_i))
		Pri_IO(io, "tmp_B: ", abs.(tmp_B))
		Pri_IO(io, "Hm_i: ", abs.(Hm_i))
		Pri_IO(io, "Hv_i1: ", Hv_i1)
		Pri_IO(io, "Hv_i0: ", Hv_i0)
	end

	@. Hv_i1 = max(Hv_i1, eps)
	@. Hv_i0 = max(Hv_i0, eps)

	if Verb > 0
		Pri_IO(io, "Hv_i1: ", Hv_i1)
		Pri_IO(io, "Hv_i0: ", Hv_i0)
	end

	Hv_i .= Hv_i0 .+ L * Hv_i1

	Mse = norm(Hm_i .- x0)^(2) / norm(x0)^(2)
	Eps = abs(Mse - Asp.Mse_O)^(2) / Mse^(2)
	Rho = dot(Hm_i, x0) / sqrt(dot(Hm_i, Hm_i) * dot(x0, x0))

	A_D = mean(Hm_i .* x0)
	A_F = mean(Hm_i.^(2))
	A_H = mean(Hv_i1)
	A_Chi = mean(Hv_i0)

	Asp.Mse_O = Mse
	Asp.Eps_O = Eps
	Asp.Rho_O = Rho
	Asp.A_D_O = A_D
	Asp.A_F_O = A_F
	Asp.A_H_O = A_H
	Asp.A_Chi_O = A_Chi

	if Verb > 0
		Pri_IO(io, "Hv_i: ", Hv_i)
		println(io, "Mse: ", Mse, ", Eps: ", Eps, ", Rho: ", Rho,
				", A_D: ", A_D, ", A_F: ", A_F, ", A_H: ", A_H, ", A_Chi: ", A_Chi
			)
	end

	println("ASP - Iter: ", t, ", Mse: ", Mse, ", Eps: ", Eps, ", Rho: ", Rho,
			", A_D: ", A_D, ", A_F: ", A_F, ", A_H: ", A_H, ", A_Chi: ", A_Chi
		)

	return Mse, Eps, Rho, A_D, A_F, A_H, A_Chi
end

Solve(Prob; kws ...) = Algo(Prob, Str_1RSB_Params(; kws ...))

function Algo(Prob, Prms::Str_1RSB_Params)
	@extract Prob: N M H x0 y
	@extract Prms: Epochs Sd Epsilon Verb Name

	Sd > 0 && Random.seed!(Sd)

	### Print utilities ###
	Df = DataFrame(
			Epoch = Int[], Mse = Float64[], Eps = Float64[], Rho = Float64[],
			A_D = Float64[], A_F = Float64[], A_H = Float64[], A_Chi = Float64[]
		)

	Asp = Setting(Prob)
	Init!(Asp, Prob, Prms)

	Ok = false

	io = 0

	if Verb > 0
		rm("./$Name-S_ASP.txt", force = true)
		io = open("./$Name-S_ASP.txt", "w")

		rm("./$Name-S_Asp_Df.bson", force = true)
		rm("./$Name-S_Asp.bson", force = true)
		rm("./$Name-S_Asp_Prms.bson", force = true)
	end

	Mse_O = 1e3
	Eps_O = 1e3
	Rho_O = 1e3

	A_D_O = 1e3
	A_F_O = 1e3
	A_H_O = 1e3
	A_Chi_O = 1e3

	for i = 1 : Epochs
		Mse, Eps, Rho, A_D, A_F, A_H, A_Chi = Iter!(Asp, i, Prms, io, x0)

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
		S_Asp_Df = deepcopy(Df)
		S_Asp = deepcopy(Asp)
		S_Asp_Prms = deepcopy(Prms)

		BSON.@save "./$Name-S_Asp_Df.bson" S_Asp_Df
		BSON.@save "./$Name-S_Asp.bson" S_Asp
		BSON.@save "./$Name-S_Asp_Prms.bson" S_Asp_Prms

		close(io)

	end

	!Ok && @warn("Not converged!")
	return Df, Asp, Ok, Prms
end

end	# Module