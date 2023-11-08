module Module_GVASP_SE

include("./Common.jl")

mutable struct Str_GVASP_SE
	D_s_z::Float64
	F_s_z::Float64
	Hv_s_z1::Float64
	Hv_s_z0::Float64
	Hv_s_z::Float64

	HD_2z::Float64
	HF_2z::Float64
	v_s_z0_inv::Float64
	v_s_z_inv::Float64

	D_s_x::Float64
	F_s_x::Float64
	Hv_s_x0::Float64
	Hv_s_x::Float64

	HD_1x::Float64
	HF_1x::Float64
	v_s_x1::Float64
	v_s_x0::Float64
	v_s_x::Float64

	D_p_x::Float64
	F_p_x::Float64
	Hv_p_x1::Float64
	Hv_p_x0::Float64
	Hv_p_x::Float64

	HD_2x::Float64
	HF_2x::Float64
	v_p_x0_inv::Float64
	v_p_x0_inv_O::Float64
	v_p_x_inv::Float64
	v_p_x_inv_O::Float64

	D_p_z::Float64
	F_p_z::Float64
	Hv_p_z0::Float64
	Hv_p_z::Float64

	HD_1z::Float64
	HF_1z::Float64
	v_p_z0_inv_O::Float64
	v_p_z0_inv::Float64
	v_p_z_inv_O::Float64
	v_p_z_inv::Float64

	Mse_O::Float64

	I_x0::Vector{Float64}
	H::Matrix{Float64}
	Ht::Matrix{Float64}
	y::Vector{Float64}
	HtH
	HHt
	C_x::Float64
	C_z::Float64
	HC_1z::Float64
end

function Setting(Prob)
	@extract Prob: N M C_x H y

	Ht = H'

	HtH = svd(Ht * H)
	HHt = svd(H * Ht)

	C_z = 1 / M * sum(HtH.S) * C_x
	HC_1z = 1 / C_z

	return Str_GVASP_SE(
			[0 for _ = 1 : 40] ...,
			zeros(N), H, Ht, y,
			HtH, HHt, C_x, C_z, HC_1z
		)
end

function Init!(Gs::Str_GVASP_SE, Prob, Prms)
	@extract Gs: I_x0
	@extract Gs: H Ht y HtH HHt C_x C_z HC_1z
	@extract Prob: a N M x0 z
	@extract Prms: L

	v1 = 1
	v0 = 1

	v = v0 + L * v1

	xIi = Init_X(H, y, Prob, Prms)
	zIi = H * xIi

	Gs.HD_2x = - mean(x0 .* xIi) / (v * C_x)
	Gs.HF_2x = - mean(
			(xIi .+ Gs.HD_2x * v * x0).^(2)
		) / v^(2)

	# println("HD_2x: ", Gs.HD_2x, ", HF_2x: ", Gs.HF_2x)

	Gs.v_p_x0_inv = 1 / v0
	Gs.v_p_x0_inv_O = Gs.v_p_x0_inv
	Gs.v_p_x_inv = 1 / v
	Gs.v_p_x_inv_O = Gs.v_p_x_inv

	Gs.HD_1z = - mean(z .* zIi) / (v * C_z)
	Gs.HF_1z = - mean(
			(zIi .+ Gs.HD_1z * v * z).^(2)
		) / v^(2)
	# println("HD_1z: ", Gs.HD_1z, ", HF_1z: ", Gs.HF_1z)

	Gs.v_p_z0_inv = 1 / v0
	Gs.v_p_z0_inv_O = Gs.v_p_z0_inv
	Gs.v_p_z_inv = 1 / v
	Gs.v_p_z_inv_O = Gs.v_p_z_inv

	I_x0 .= xIi

	Gs.Mse_O = 1e3
end

function Iter!(Gs::Str_GVASP_SE, t, Prob, Prms, io, x0)
	@extract Gs: I_x0
	@extract Gs: H Ht y HtH HHt C_x C_z HC_1z
	@extract Prob: a N M
	@extract Prms: L La Mes va EPS Verb

	if Verb > 0
		println(io, "")
		println(io, "t: ", t)
	end

	v_p_z0 = 1 / Gs.v_p_z0_inv
	v_p_z = 1 / Gs.v_p_z_inv

	v_p_z1 = 1 / L * (v_p_z - v_p_z0)

	if Verb > 0
		println(io, "C_x: ", C_x)
		println(io, "C_z: ", C_z)
		println(io, "HC_1z: ", HC_1z)

		println(io, "v_p_z0: ", v_p_z0)
		println(io, "v_p_z: ", v_p_z)
		println(io, "v_p_z1: ", v_p_z1)
	end

	v_p_z1 = max(v_p_z1, EPS)

	Out_Res = O_SE_1RSB(HC_1z, Gs.HD_1z, Gs.HF_1z, v_p_z1, v_p_z0, L, va, EPS)
	Gs.D_s_z = Out_Res.DZ
	Gs.F_s_z = Out_Res.FZ
	Gs.Hv_s_z1 = Out_Res.HZ
	Gs.Hv_s_z0 = Out_Res.GZsFZ

	if Verb > 0
		println(io, "D_s_z: ", Gs.D_s_z)
		println(io, "F_s_z: ", Gs.F_s_z)

		println(io, "Hv_s_z1: ", Gs.Hv_s_z1)
		println(io, "Hv_s_z0: ", Gs.Hv_s_z0)
	end

	Gs.Hv_s_z1 = max(Gs.Hv_s_z1, EPS)
	Gs.Hv_s_z0 = max(Gs.Hv_s_z0, EPS)

	Gs.Hv_s_z = Gs.Hv_s_z0 + L * Gs.Hv_s_z1

	if Verb > 0
		println(io, "Hv_s_z: ", Gs.Hv_s_z)
	end

	Gs.v_s_z0_inv = 1 / Gs.Hv_s_z0 - Gs.v_p_z0_inv
	Gs.v_s_z_inv = 1 / Gs.Hv_s_z - Gs.v_p_z_inv

	if Verb > 0
		println(io, "v_s_z0_inv: ", Gs.v_s_z0_inv)
		println(io, "v_s_z_inv: ", Gs.v_s_z_inv)
	end

	v_s_z0 = 1 / Gs.v_s_z0_inv
	v_s_z = 1 / Gs.v_s_z_inv

	if Verb > 0
		println(io, "v_s_z0: ", v_s_z0)
		println(io, "v_s_z: ", v_s_z)
	end

	v_s_z0 = max(v_s_z0, EPS)
	v_s_z = max(v_s_z, EPS)

	if Verb > 0
		tmp_v_s_z0 = v_s_z0 + EPS
		Res_v_s_z0 = (v_s_z >= tmp_v_s_z0)
		println(io, "Res_v_s_z0: ", Res_v_s_z0)
	end

	v_s_z = max(v_s_z, v_s_z0 + EPS)

	Gs.v_s_z0_inv = 1 / v_s_z0
	Gs.v_s_z_inv = 1 / v_s_z

	Gs.HD_2z = - (
		Gs.D_s_z / (C_z * Gs.Hv_s_z) + Gs.HD_1z
		)
	Gs.HF_2z = - (
		Gs.F_s_z / Gs.Hv_s_z^(2) - Gs.D_s_z^(2) / (C_z * Gs.Hv_s_z^(2)) + Gs.HF_1z
	)

	if Verb > 0
		println(io, "HD_2z: ", Gs.HD_2z)
		println(io, "HF_2z: ", Gs.HF_2z)
	end

	if Gs.HD_2z > 0.
		Gs.HD_2z = - EPS
	end

	if Gs.HF_2z > 0.
		Gs.HF_2z = - EPS
	end

	if Verb > 0
		println(io, "HD_2z: ", Gs.HD_2z)
		println(io, "HF_2z: ", Gs.HF_2z)
	end

	Gs.Hv_s_x0 = 1 / N * sum(
			1 ./ (Gs.v_p_x0_inv .+ Gs.v_s_z0_inv * HtH.S)
		)
	Gs.Hv_s_x = 1 / N * sum(
			1 ./ (Gs.v_p_x_inv .+ Gs.v_s_z_inv * HtH.S)
		)

	if Verb > 0
		println(io, "Hv_s_x0: ", Gs.Hv_s_x0)
		println(io, "Hv_s_x: ", Gs.Hv_s_x)
	end

	Gs.Hv_s_x0 = max(Gs.Hv_s_x0, EPS)
	Gs.Hv_s_x = max(Gs.Hv_s_x, EPS)

	if Verb > 0
		tmp_Hv_s_x0 = Gs.Hv_s_x0 + EPS
		Res_Hv_s_x0 = (Gs.Hv_s_x >= tmp_Hv_s_x0)
		println(io, "Res_Hv_s_x0: ", Res_Hv_s_x0)
	end

	Gs.Hv_s_x = max(Gs.Hv_s_x, Gs.Hv_s_x0 + EPS)

	Gs.D_s_x = - 1 / N * C_x * sum(
			(Gs.HD_2x .+ Gs.HD_2z * HtH.S) ./ (Gs.v_p_x_inv .+ Gs.v_s_z_inv * HtH.S)
		)
	Gs.F_s_x = 1 / N * C_x * sum(
		(Gs.HD_2x .+ Gs.HD_2z * HtH.S).^(2) ./ (Gs.v_p_x_inv .+ Gs.v_s_z_inv * HtH.S).^(2)
	) - 1 / N * sum(
		(Gs.HF_2x .+ Gs.HF_2z * HtH.S) ./ (Gs.v_p_x_inv .+ Gs.v_s_z_inv * HtH.S).^(2)
	)

	if Verb > 0
		println(io, "D_s_x: ", Gs.D_s_x)
		println(io, "F_s_x: ", Gs.F_s_x)
	end

	Gs.v_s_x0 = Gs.Hv_s_x0 / (1 - Gs.Hv_s_x0 * Gs.v_p_x0_inv)
	Gs.v_s_x = Gs.Hv_s_x / (1 - Gs.Hv_s_x * Gs.v_p_x_inv)

	if Verb > 0
		println(io, "v_s_x0: ", Gs.v_s_x0)
		println(io, "v_s_x: ", Gs.v_s_x)
	end

	Gs.v_s_x0 = max(Gs.v_s_x0, EPS)
	Gs.v_s_x = max(Gs.v_s_x, EPS)

	if Verb > 0
		tmp_v_s_x0 = Gs.v_s_x0 + EPS
		Res_v_s_x0 = (Gs.v_s_x >= tmp_v_s_x0)
		println(io, "Res_v_s_x0: ", Res_v_s_x0)
	end

	Gs.v_s_x = max(Gs.v_s_x, Gs.v_s_x0 + EPS)

	Gs.v_s_x1 = 1 / L * (Gs.v_s_x - Gs.v_s_x0)

	if Verb > 0
		println(io, "v_s_x1: ", Gs.v_s_x1)
	end

	Gs.v_s_x1 = max(Gs.v_s_x1, EPS)

	Gs.HD_1x = - (
			Gs.D_s_x / (C_x * Gs.Hv_s_x) + Gs.HD_2x
		)
	Gs.HF_1x = - (
			Gs.F_s_x / Gs.Hv_s_x^(2) - Gs.D_s_x^(2) / (C_x * Gs.Hv_s_x^(2)) + Gs.HF_2x
		)

	if Verb > 0
		println(io, "HD_1x: ", Gs.HD_1x)
		println(io, "HF_1x: ", Gs.HF_1x)
	end

	if Gs.HD_1x > 0.
		Gs.HD_1x = - EPS
	end

	if Gs.HF_1x > 0.
		Gs.HF_1x = - EPS
	end

	if Verb > 0
		println(io, "HD_1x: ", Gs.HD_1x)
		println(io, "HF_1x: ", Gs.HF_1x)
	end

	In_Res = I_SE_1RSB(Gs.HD_1x, Gs.HF_1x, Gs.v_s_x1, Gs.v_s_x0, L, La, EPS, C_x)
	Gs.D_p_x = In_Res.DX
	Gs.F_p_x = In_Res.FX
	Gs.Hv_p_x1 = In_Res.HX
	Gs.Hv_p_x0 = In_Res.GXsFX

	if Verb > 0
		println(io, "D_p_x: ", Gs.D_p_x)
		println(io, "F_p_x: ", Gs.F_p_x)
		println(io, "Hv_p_x1: ", Gs.Hv_p_x1)
		println(io, "Hv_p_x0: ", Gs.Hv_p_x0)
	end

	Gs.Hv_p_x1 = max(Gs.Hv_p_x1, EPS)
	Gs.Hv_p_x0 = max(Gs.Hv_p_x0, EPS)

	Gs.Hv_p_x = Gs.Hv_p_x0 + L * Gs.Hv_p_x1

	Mse = (C_x + Gs.F_p_x - 2 * Gs.D_p_x) / C_x
	Eps = abs(Mse - Gs.Mse_O)^(2) / Mse^(2)
	Rho = Gs.D_p_x / sqrt(Gs.F_p_x * C_x)

	A_D = Gs.D_p_x
	A_F = Gs.F_p_x
	A_H = Gs.Hv_p_x1
	A_GsF = Gs.Hv_p_x0

	Gs.Mse_O = Mse

	if Verb > 0
		println(io, "Hv_p_x: ", Gs.Hv_p_x)
		println(io, "Mse: ", Mse, ", Eps: ", Eps, ", Rho: ", Rho,
				", A_D: ", A_D, ", A_F: ", A_F, ", A_H: ", A_H, ", A_GsF: ", A_GsF
			)
	end

	println("GVASP-SE - Iter: ", t, ", Mse: ", Mse, ", Eps: ", Eps, ", Rho: ", Rho,
			", A_D: ", A_D, ", A_F: ", A_F, ", A_H: ", A_H, ", A_GsF: ", A_GsF
		)

	Gs.v_p_x0_inv = (Gs.v_s_x0 - Gs.Hv_p_x0) / (Gs.v_s_x0 * Gs.Hv_p_x0)
	Gs.v_p_x_inv = (Gs.v_s_x - Gs.Hv_p_x) / (Gs.v_s_x * Gs.Hv_p_x)

	if Verb > 0
		println(io, "v_p_x0_inv: ", Gs.v_p_x0_inv)
		println(io, "v_p_x_inv: ", Gs.v_p_x_inv)
	end

	if Gs.v_p_x0_inv < EPS
		if Verb > 0
			println(io, "Res_v_p_x0_inv: !!!")
		end

		Gs.v_p_x0_inv = Gs.v_p_x0_inv_O
	end

	if Gs.v_p_x_inv < EPS
		if Verb > 0
			println(io, "Res_v_p_x_inv: !!!")
		end

		Gs.v_p_x_inv = Gs.v_p_x_inv_O
	end

	Gs.v_p_x0_inv = Mes * Gs.v_p_x0_inv + (1 - Mes) * Gs.v_p_x0_inv_O
	Gs.v_p_x0_inv_O = Gs.v_p_x0_inv
	Gs.v_p_x_inv = Mes * Gs.v_p_x_inv + (1 - Mes) * Gs.v_p_x_inv_O
	Gs.v_p_x_inv_O = Gs.v_p_x_inv

	if Verb > 0
		println(io, "v_p_x0_inv: ", Gs.v_p_x0_inv)
		println(io, "v_p_x_inv: ", Gs.v_p_x_inv)
	end

	T_v_p_x0 = 1 / Gs.v_p_x0_inv
	T_v_p_x = 1 / Gs.v_p_x_inv

	if Verb > 0
		println(io, "T_v_p_x0: ", T_v_p_x0)
		println(io, "T_v_p_x: ", T_v_p_x)
	end

	if Verb > 0
		tmp_T_v_p_x0 = T_v_p_x0 + EPS
		Res_T_v_p_x0 = (T_v_p_x >= tmp_T_v_p_x0)
		println(io, "Res_T_v_p_x0: ", Res_T_v_p_x0)
	end

	T_v_p_x = max(T_v_p_x, T_v_p_x0 + EPS)

	Gs.v_p_x_inv = 1 / T_v_p_x
	Gs.v_p_x_inv_O = Gs.v_p_x_inv

	Gs.HD_2x = - (
			Gs.D_p_x / (C_x * Gs.Hv_p_x) + Gs.HD_1x
		)
	Gs.HF_2x = - (
		Gs.F_p_x / Gs.Hv_p_x^(2) - Gs.D_p_x^(2) / (C_x * Gs.Hv_p_x^(2)) + Gs.HF_1x
	)

	if Verb > 0
		println(io, "HD_2x: ", Gs.HD_2x)
		println(io, "HF_2x: ", Gs.HF_2x)
	end

	if Gs.HD_2x > 0.
		Gs.HD_2x = - EPS
	end

	if Gs.HF_2x > 0.
		Gs.HF_2x = - EPS
	end

	if Verb > 0
		println(io, "HD_2x: ", Gs.HD_2x)
		println(io, "HF_2x: ", Gs.HF_2x)
	end

	Gs.Hv_p_z0 = 1 / M * sum(
			HHt.S ./ (Gs.v_p_x0_inv .+ Gs.v_s_z0_inv * HHt.S)
		)
	Gs.Hv_p_z = 1 / M * sum(
			HHt.S ./ (Gs.v_p_x_inv .+ Gs.v_s_z_inv * HHt.S)
		)

	if Verb > 0
		println(io, "Hv_p_z0: ", Gs.Hv_p_z0)
		println(io, "Hv_p_z: ", Gs.Hv_p_z)
	end

	Gs.Hv_p_z0 = max(Gs.Hv_p_z0, EPS)
	Gs.Hv_p_z = max(Gs.Hv_p_z, EPS)

	if Verb > 0
		tmp_Hv_p_z0 = Gs.Hv_p_z0 + EPS
		Res_Hv_p_z0 = (Gs.Hv_p_z >= tmp_Hv_p_z0)
		println(io, "Res_Hv_p_z0: ", Res_Hv_p_z0)
	end

	Gs.Hv_p_z = max(Gs.Hv_p_z, Gs.Hv_p_z0 + EPS)

	Gs.D_p_z = - 1 / M * C_x * sum(
			HHt.S .* (Gs.HD_2x .+ Gs.HD_2z * HHt.S) ./ (Gs.v_p_x_inv .+ Gs.v_s_z_inv * HHt.S)
		)
	Gs.F_p_z = 1 / M * C_x * sum(
		HHt.S .* (Gs.HD_2x .+ Gs.HD_2z * HHt.S).^(2) ./ (Gs.v_p_x_inv .+ Gs.v_s_z_inv * HHt.S).^(2)
	) - 1 / M * sum(
		HHt.S .* (Gs.HF_2x .+ Gs.HF_2z * HHt.S) ./ (Gs.v_p_x_inv .+ Gs.v_s_z_inv * HHt.S).^(2)
	)

	if Verb > 0
		println(io, "D_p_z: ", Gs.D_p_z)
		println(io, "F_p_z: ", Gs.F_p_z)
	end

	Gs.v_p_z0_inv = 1 / Gs.Hv_p_z0 - Gs.v_s_z0_inv
	Gs.v_p_z_inv = 1 / Gs.Hv_p_z - Gs.v_s_z_inv

	if Verb > 0
		println(io, "v_p_z0_inv: ", Gs.v_p_z0_inv)
		println(io, "v_p_z_inv: ", Gs.v_p_z_inv)
	end

	if Gs.v_p_z0_inv < 0
		if Verb > 0
			println(io, "Res_v_p_z0_inv: !!!")
		end

		Gs.v_p_z0_inv = Gs.v_p_z0_inv_O
	end

	if Gs.v_p_z_inv < 0
		if Verb > 0
			println(io, "Res_v_p_z_inv: !!!")
		end

		Gs.v_p_z_inv = Gs.v_p_z_inv_O
	end

	Gs.v_p_z0_inv = Mes * Gs.v_p_z0_inv + (1 - Mes) * Gs.v_p_z0_inv_O
	Gs.v_p_z0_inv_O = Gs.v_p_z0_inv
	Gs.v_p_z_inv = Mes * Gs.v_p_z_inv + (1 - Mes) * Gs.v_p_z_inv_O
	Gs.v_p_z_inv_O = Gs.v_p_z_inv

	if Verb > 0
		println(io, "v_p_z0_inv: ", Gs.v_p_z0_inv)
		println(io, "v_p_z_inv: ", Gs.v_p_z_inv)
	end

	T_v_p_z0 = 1 / Gs.v_p_z0_inv
	T_v_p_z = 1 / Gs.v_p_z_inv

	if Verb > 0
		println(io, "T_v_p_z0: ", T_v_p_z0)
		println(io, "T_v_p_z: ", T_v_p_z)
	end

	if Verb > 0
		tmp_T_v_p_z0 = T_v_p_z0 + EPS
		Res_T_v_p_z0 = (T_v_p_z >= tmp_T_v_p_z0)
		println(io, "Res_T_v_p_z0: ", Res_T_v_p_z0)
	end

	T_v_p_z = max(T_v_p_z, T_v_p_z0 + EPS)

	Gs.v_p_z_inv = 1 / T_v_p_z
	Gs.v_p_z_inv_O = Gs.v_p_z_inv

	Gs.HD_1z = - (
			Gs.D_p_z / (C_z * Gs.Hv_p_z) + Gs.HD_2z
		)
	Gs.HF_1z = - (
			Gs.F_p_z / Gs.Hv_p_z^(2) - Gs.D_p_z^(2) / (C_z * Gs.Hv_p_z^(2)) + Gs.HF_2z
		)

	if Verb > 0
		println(io, "HD_1z: ", Gs.HD_1z)
		println(io, "HF_1z: ", Gs.HF_1z)
	end

	if Gs.HD_1z > 0.
		Gs.HD_1z = - EPS
	end

	if Gs.HF_1z > 0.
		Gs.HF_1z = - EPS
	end

	if Verb > 0
		println(io, "HD_1z: ", Gs.HD_1z)
		println(io, "HF_1z: ", Gs.HF_1z)
	end

	return Mse, Eps, Rho, A_D, A_F, A_H, A_GsF
end

Solve(Prob; kws ...) = SE(Prob, Str_1RSB_Params(; kws ...))

function SE(Prob, Prms::Str_1RSB_Params)
	@extract Prob: N M H x0 y
	@extract Prms: Epochs La Drop_La Sd Epsilon Verb Name

	Sd > 0 && Random.seed!(Sd)

	### Print utilities ###
	Df = DataFrame(
			Epoch = Int[], Mse = Float64[], Eps = Float64[], Rho = Float64[],
			A_D = Float64[], A_F = Float64[], A_H = Float64[], A_GsF = Float64[]
		)

	Gs = Setting(Prob)
	Init!(Gs, Prob, Prms)

	Ok = false

	io = 0

	if Verb > 0
		rm("./$Name-S_GVASP_SE.txt", force = true)
		io = open("./$Name-S_GVASP_SE.txt", "w")

		rm("./$Name-S_Gs_Df.bson", force = true)
		rm("./$Name-S_Gs.bson", force = true)
		rm("./$Name-S_Gs_Prms.bson", force = true)
	end

	Mse_O = 1e3
	Eps_O = 1e3
	Rho_O = 1e3

	A_D_O = 1e3
	A_F_O = 1e3
	A_H_O = 1e3
	A_GsF_O = 1e3

	for i = 1 : Epochs
		Mse, Eps, Rho, A_D, A_F, A_H, A_GsF = Iter!(Gs, i, Prob, Prms, io, x0)

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
			(La <= 0 || !Drop_La) && break
			println("# Set La = 0")
			La = 0
		end
	end

	Num = size(Df, 1) + 1

	for i = Num : Epochs
		Report!(Df, i, Mse_O, Eps_O, Rho_O, A_D_O, A_F_O, A_H_O, A_GsF_O)
	end

	if Verb > 0
		S_Gs_Df = deepcopy(Df)
		S_Gs = deepcopy(Gs)
		S_Gs_Prms = deepcopy(Prms)

		BSON.@save "./$Name-S_Gs_Df.bson" S_Gs_Df
		BSON.@save "./$Name-S_Gs.bson" S_Gs
		BSON.@save "./$Name-S_Gs_Prms.bson" S_Gs_Prms

		close(io)
	end

	!Ok && @warn("Not converged!")
	return Df, Gs, Ok, Prms
end

end	# Module