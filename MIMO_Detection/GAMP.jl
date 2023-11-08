module Module_GAMP

include("./Common.jl")

mutable struct Str_GAMP
	Z_a_O::Vector{Float64}
	Z_a::Vector{Float64}
	V_a_O::Vector{Float64}
	V_a::Vector{Float64}

	Tz_a::Vector{Float64}
	Tv_a::Vector{Float64}

	s_a::Vector{Float64}
	t_a::Vector{Float64}

	Tm_i::Vector{Float64}
	Tv_i::Vector{Float64}

	Hm_i::Vector{Float64}
	Hv_i::Vector{Float64}

	I_x0::Vector{Float64}

	Mse_O::Float64

	H::Matrix{Float64}
	y::Vector{Float64}
end

function Setting(Prob)
	@extract Prob: N M H y

	return Str_GAMP(
			[zeros(M) for _ = 1 : 8] ...,
			[zeros(N) for _ = 1 : 5] ...,
			0, H, y
		)
end

function Init!(Gamp::Str_GAMP, Prob, Prms)
	@extract Gamp: Z_a_O V_a_O s_a
	@extract Gamp: Hm_i Hv_i I_x0
	@extract Gamp: H y

	v = 1

	Z_a_O .= 0
	V_a_O .= v

	s_a .= 0

	xIi = Init_X(H, y, Prob, Prms)

	Hm_i .= xIi

	Hv_i .= v

	I_x0 .= xIi

	Gamp.Mse_O = 1e3
end

function Iter!(Gamp::Str_GAMP, t, Prms, io, x0)
	@extract Gamp: Z_a_O Z_a V_a_O V_a Tz_a Tv_a s_a t_a
	@extract Gamp: Tm_i Tv_i Hm_i Hv_i
	@extract Gamp: H y
	@extract Prms: Mes Ax_F vw_F EPS Verb

	if Verb > 0
		println(io, "")
		println(io, "t: ", t)
	end

	Ht = H'
	H2 = abs.(H).^(2)
	H2t = H2'

	V_a .= H2 * Hv_i

	if Verb > 0
		Pri_IO(io, "V_a: ", V_a)
	end

	Z_a .= H * Hm_i .- s_a .* V_a

	if Verb > 0
		Pri_IO(io, "Z_a: ", abs.(Z_a))
	end

	V_a .= Mes * V_a .+ (1 - Mes) * V_a_O
	V_a_O .= V_a

	Z_a .= Mes * Z_a .+ (1 - Mes) * Z_a_O
	Z_a_O .= Z_a

	Out_Res = O_Est_RS(y, Z_a, V_a, vw_F, EPS)
	Tz_a .= Out_Res.Hm
	Tv_a .= Out_Res.Hv

	if Verb > 0
		Pri_IO(io, "Tz_a: ", abs.(Tz_a))
		Pri_IO(io, "Tv_a: ", Tv_a)
	end

	@. Tv_a = max(Tv_a, EPS)

	s_a .= (Tz_a .- Z_a) ./ V_a
	t_a .= (V_a .- Tv_a) ./ abs.(V_a).^(2)

	if Verb > 0
		Pri_IO(io, "Tv_a: ", Tv_a)
		Pri_IO(io, "s_a: ", abs.(s_a))
		Pri_IO(io, "t_a: ", t_a)
	end

	@. t_a = max(t_a, EPS)

	Tv_i .= 1 ./ (H2t * t_a)

	if Verb > 0
		tmp_A = H2t * t_a

		Pri_IO(io, "Tv_i: ", Tv_i)
		Pri_IO(io, "tmp_A: ", tmp_A)
	end

	@. Tv_i = max(Tv_i, EPS)

	Tm_i .= Hm_i .+ Tv_i .* (Ht * s_a)

	In_Res = I_Est_RS(Tm_i, Tv_i, Ax_F, EPS)
	Hm_i .= In_Res.Hm
	Hv_i .= In_Res.Hv

	if Verb > 0
		tmp_B = Tm_i ./ Tv_i

		Pri_IO(io, "Tm_i: ", abs.(Tm_i))
		Pri_IO(io, "tmp_B: ", abs.(tmp_B))
		Pri_IO(io, "Hm_i: ", abs.(Hm_i))
		Pri_IO(io, "Hv_i: ", Hv_i)
	end

	@. Hv_i = max(Hv_i, EPS)

	Mse = norm(Hm_i .- x0)^(2) / norm(x0)^(2)
	Eps = abs(Mse - Gamp.Mse_O)^(2) / Mse^(2)
	Rho = dot(Hm_i, x0) / sqrt(dot(Hm_i, Hm_i) * dot(x0, x0))

	A_D = mean(Hm_i .* x0)
	A_F = mean(Hm_i.^(2))
	A_H = 0
	A_GsF = mean(Hv_i)

	Gamp.Mse_O = Mse

	if Verb > 0
		Pri_IO(io, "Hv_i: ", Hv_i)
		println(io, "Mse: ", Mse, ", Eps: ", Eps, ", Rho: ", Rho,
				", A_D: ", A_D, ", A_F: ", A_F, ", A_H: ", A_H, ", A_GsF: ", A_GsF
			)
	end

	println("GAMP - Iter: ", t, ", Mse: ", Mse, ", Eps: ", Eps, ", Rho: ", Rho,
			", A_D: ", A_D, ", A_F: ", A_F, ", A_H: ", A_H, ", A_GsF: ", A_GsF
		)

	return Mse, Eps, Rho, A_D, A_F, A_H, A_GsF
end

Solve(Prob; kws ...) = Algo(Prob, Str_RS_Params(; kws ...))

function Algo(Prob, Prms::Str_RS_Params)
	@extract Prob: N M H x0 y
	@extract Prms: Epochs Sd Epsilon Verb Name

	Sd > 0 && Random.seed!(Sd)

	### Print utilities ###
	Df = DataFrame(
			Epoch = Int[], Mse = Float64[], Eps = Float64[], Rho = Float64[],
			A_D = Float64[], A_F = Float64[], A_H = Float64[], A_GsF = Float64[]
		)

	Gamp = Setting(Prob)
	Init!(Gamp, Prob, Prms)

	Ok = false

	io = 0

	if Verb > 0
		rm("./$Name-S_GAMP.txt", force = true)
		io = open("./$Name-S_GAMP.txt", "w")

		rm("./$Name-S_Gamp_Df.bson", force = true)
		rm("./$Name-S_Gamp.bson", force = true)
		rm("./$Name-S_Gamp_Prms.bson", force = true)
	end

	Mse_O = 1e3
	Eps_O = 1e3
	Rho_O = 1e3

	A_D_O = 1e3
	A_F_O = 1e3
	A_H_O = 1e3
	A_GsF_O = 1e3

	for i = 1 : Epochs
		Mse, Eps, Rho, A_D, A_F, A_H, A_GsF = Iter!(Gamp, i, Prms, io, x0)

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
		S_Gamp_Df = deepcopy(Df)
		S_Gamp = deepcopy(Gamp)
		S_Gamp_Prms = deepcopy(Prms)

		BSON.@save "./$Name-S_Gamp_Df.bson" S_Gamp_Df
		BSON.@save "./$Name-S_Gamp.bson" S_Gamp
		BSON.@save "./$Name-S_Gamp_Prms.bson" S_Gamp_Prms

		close(io)

	end

	!Ok && @warn("Not converged!")
	return Df, Gamp, Ok, Prms
end

end	# Module