module Module_VASP_SE

include("./Common.jl")

mutable struct Str_VASP_SE
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
	Eps_O::Float64
	Rho_O::Float64
	A_D_O::Float64
	A_F_O::Float64
	A_H_O::Float64
	A_Chi_O::Float64

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
	@extract Prob: N M Ax_T Cx_T H y

	C_x = Cx_T + Ax_T^(2) + Ax_T * sqrt(Cx_T) * (
			pdf(
				Normal(0, 1), Ax_T / sqrt(Cx_T)
			) / cdf(
				Normal(0, 1), Ax_T / sqrt(Cx_T)
			)
		)

	println("C_x: ", C_x)

	Ht = H'

	HtH = svd(Ht * H)
	HHt = svd(H * Ht)

	C_z = 1 / M * sum(HtH.S) * C_x
	HC_1z = 1 / C_z

	println("C_z: ", C_z)

	# xasxasxas

	return Str_VASP_SE(
			[0 for _ = 1 : 46] ...,
			zeros(N), H, Ht, y,
			HtH, HHt, C_x, C_z, HC_1z
		)
end

function Init!(Vs::Str_VASP_SE, Prob, Prms)
	@extract Vs: I_x0
	@extract Vs: H Ht y HtH HHt C_x C_z HC_1z
	@extract Prob: a N M x0 z
	@extract Prms: L

	v1 = 1
	v0 = 1

	v = v0 + L * v1

	xIi = Init_X(H, y, Prob, Prms)
	zIi = H * xIi

	Vs.HD_2x = mean(x0 .* xIi) / (v * C_x)
	Vs.HF_2x = mean(
			(xIi .- v * Vs.HD_2x * x0).^(2)
		) / v^(2)

	println("HD_2x: ", Vs.HD_2x, ", HF_2x: ", Vs.HF_2x)

	Vs.v_p_x0_inv = 1 / v0
	Vs.v_p_x0_inv_O = Vs.v_p_x0_inv
	Vs.v_p_x_inv = 1 / v
	Vs.v_p_x_inv_O = Vs.v_p_x_inv

	Vs.HD_1z = mean(z .* zIi) / (v * C_z)
	Vs.HF_1z = mean(
			(zIi .- v * Vs.HD_1z * z).^(2)
		) / v^(2)
	println("HD_1z: ", Vs.HD_1z, ", HF_1z: ", Vs.HF_1z)

	Vs.v_p_z0_inv = 1 / v0
	Vs.v_p_z0_inv_O = Vs.v_p_z0_inv
	Vs.v_p_z_inv = 1 / v
	Vs.v_p_z_inv_O = Vs.v_p_z_inv

	I_x0 .= xIi

	Vs.Mse_O = 1e3
	Vs.Eps_O = 1e3
	Vs.Rho_O = 1e3
	Vs.A_D_O = 1e3
	Vs.A_F_O = 1e3
	Vs.A_H_O = 1e3
	Vs.A_Chi_O = 1e3
end

function Iter!(Vs::Str_VASP_SE, t, Prob, Prms, io, x0)
	@extract Vs: I_x0
	@extract Vs: H Ht y HtH HHt C_x C_z HC_1z
	@extract Prob: a N M Ax_T Cx_T vw_T
	@extract Prms: L Mes Ax_F Cx_F vw_F eps Verb

	if Verb > 0
		println(io, "")
		println(io, "t: ", t)
	end

	v_p_z0 = 1 / Vs.v_p_z0_inv
	v_p_z = 1 / Vs.v_p_z_inv

	v_p_z1 = 1 / L * (v_p_z - v_p_z0)

	if Verb > 0
		println(io, "C_x: ", C_x)
		println(io, "C_z: ", C_z)
		println(io, "HC_1z: ", HC_1z)

		println(io, "v_p_z0: ", v_p_z0)
		println(io, "v_p_z: ", v_p_z)
		println(io, "v_p_z1: ", v_p_z1)
	end

	v_p_z1 = max(v_p_z1, eps)

	if Verb > 0
		println(io, "v_p_z1: ", v_p_z1)
	end

	Out_Res = O_SE_1RSB(HC_1z, Vs.HD_1z, Vs.HF_1z, v_p_z1, v_p_z0, L, vw_T, vw_F, eps)
	Vs.D_s_z = Out_Res.DZ
	Vs.F_s_z = Out_Res.FZ
	Vs.Hv_s_z1 = Out_Res.HZ
	Vs.Hv_s_z0 = Out_Res.ChiZ

	if Verb > 0
		println(io, "D_s_z: ", Vs.D_s_z)
		println(io, "F_s_z: ", Vs.F_s_z)

		println(io, "Hv_s_z1: ", Vs.Hv_s_z1)
		println(io, "Hv_s_z0: ", Vs.Hv_s_z0)
	end

	Vs.Hv_s_z1 = max(Vs.Hv_s_z1, eps)
	Vs.Hv_s_z0 = max(Vs.Hv_s_z0, eps)

	if Verb > 0
		println(io, "Hv_s_z1: ", Vs.Hv_s_z1)
		println(io, "Hv_s_z0: ", Vs.Hv_s_z0)
	end

	Vs.Hv_s_z = Vs.Hv_s_z0 + L * Vs.Hv_s_z1

	if Verb > 0
		println(io, "Hv_s_z: ", Vs.Hv_s_z)
	end

	Vs.v_s_z0_inv = 1 / Vs.Hv_s_z0 - Vs.v_p_z0_inv
	Vs.v_s_z_inv = 1 / Vs.Hv_s_z - Vs.v_p_z_inv

	if Verb > 0
		println(io, "v_s_z0_inv: ", Vs.v_s_z0_inv)
		println(io, "v_s_z_inv: ", Vs.v_s_z_inv)
	end

	v_s_z0 = 1 / Vs.v_s_z0_inv
	v_s_z = 1 / Vs.v_s_z_inv

	if Verb > 0
		println(io, "v_s_z0: ", v_s_z0)
		println(io, "v_s_z: ", v_s_z)
	end

	v_s_z0 = max(v_s_z0, eps)
	v_s_z = max(v_s_z, eps)

	if Verb > 0
		println(io, "v_s_z0: ", v_s_z0)
		println(io, "v_s_z: ", v_s_z)
	end

	if Verb > 0
		tmp_v_s_z0 = v_s_z0 + eps
		Res_v_s_z0 = (v_s_z >= tmp_v_s_z0)
		println(io, "Res_v_s_z0: ", Res_v_s_z0)
	end

	v_s_z = max(v_s_z, v_s_z0 + eps)

	if Verb > 0
		println(io, "v_s_z: ", v_s_z)
	end

	Vs.v_s_z0_inv = 1 / v_s_z0
	Vs.v_s_z_inv = 1 / v_s_z

	if Verb > 0
		println(io, "v_s_z0_inv: ", Vs.v_s_z0_inv)
		println(io, "v_s_z_inv: ", Vs.v_s_z_inv)
	end

	Vs.HD_2z = Vs.D_s_z / (C_z * Vs.Hv_s_z) - Vs.HD_1z
	Vs.HF_2z = Vs.F_s_z / Vs.Hv_s_z^(2) -
		Vs.D_s_z^(2) / (C_z * Vs.Hv_s_z^(2)) - Vs.HF_1z

	if Verb > 0
		println(io, "HD_2z: ", Vs.HD_2z)
		println(io, "HF_2z: ", Vs.HF_2z)
	end

	if Vs.HD_2z < 0.
		Vs.HD_2z = eps
	end

	if Vs.HF_2z < 0.
		Vs.HF_2z = eps
	end

	if Verb > 0
		println(io, "HD_2z: ", Vs.HD_2z)
		println(io, "HF_2z: ", Vs.HF_2z)
	end

	Vs.Hv_s_x0 = 1 / N * sum(
			1 ./ (
				Vs.v_p_x0_inv .+ Vs.v_s_z0_inv * HtH.S
			)
		)
	Vs.Hv_s_x = 1 / N * sum(
			1 ./ (
				Vs.v_p_x_inv .+ Vs.v_s_z_inv * HtH.S
			)
		)

	if Verb > 0
		println(io, "Hv_s_x0: ", Vs.Hv_s_x0)
		println(io, "Hv_s_x: ", Vs.Hv_s_x)
	end

	Vs.Hv_s_x0 = max(Vs.Hv_s_x0, eps)
	Vs.Hv_s_x = max(Vs.Hv_s_x, eps)

	if Verb > 0
		println(io, "Hv_s_x0: ", Vs.Hv_s_x0)
		println(io, "Hv_s_x: ", Vs.Hv_s_x)
	end

	if Verb > 0
		tmp_Hv_s_x0 = Vs.Hv_s_x0 + eps
		Res_Hv_s_x0 = (Vs.Hv_s_x >= tmp_Hv_s_x0)
		println(io, "Res_Hv_s_x0: ", Res_Hv_s_x0)
	end

	Vs.Hv_s_x = max(Vs.Hv_s_x, Vs.Hv_s_x0 + eps)

	if Verb > 0
		println(io, "Hv_s_x: ", Vs.Hv_s_x)
	end

	Vs.D_s_x = C_x / N * sum(
			(
				Vs.HD_2x .+ Vs.HD_2z * HtH.S
			) ./ (
				Vs.v_p_x_inv .+ Vs.v_s_z_inv * HtH.S
			)
		)
	Vs.F_s_x = C_x / N * sum(
			(
				Vs.HD_2x .+ Vs.HD_2z * HtH.S
			).^(2) ./ (
				Vs.v_p_x_inv .+ Vs.v_s_z_inv * HtH.S
			).^(2)
		) + 1 / N * sum(
			(
				Vs.HF_2x .+ Vs.HF_2z * HtH.S
			) ./ (
				Vs.v_p_x_inv .+ Vs.v_s_z_inv * HtH.S
			).^(2)
		)

	if Verb > 0
		println(io, "D_s_x: ", Vs.D_s_x)
		println(io, "F_s_x: ", Vs.F_s_x)
	end

	Vs.v_s_x0 = Vs.Hv_s_x0 / (1 - Vs.Hv_s_x0 * Vs.v_p_x0_inv)
	Vs.v_s_x = Vs.Hv_s_x / (1 - Vs.Hv_s_x * Vs.v_p_x_inv)

	if Verb > 0
		println(io, "v_s_x0: ", Vs.v_s_x0)
		println(io, "v_s_x: ", Vs.v_s_x)
	end

	Vs.v_s_x0 = max(Vs.v_s_x0, eps)
	Vs.v_s_x = max(Vs.v_s_x, eps)

	if Verb > 0
		println(io, "v_s_x0: ", Vs.v_s_x0)
		println(io, "v_s_x: ", Vs.v_s_x)
	end

	if Verb > 0
		tmp_v_s_x0 = Vs.v_s_x0 + eps
		Res_v_s_x0 = (Vs.v_s_x >= tmp_v_s_x0)
		println(io, "Res_v_s_x0: ", Res_v_s_x0)
	end

	Vs.v_s_x = max(Vs.v_s_x, Vs.v_s_x0 + eps)

	if Verb > 0
		println(io, "v_s_x: ", Vs.v_s_x)
	end

	Vs.v_s_x1 = 1 / L * (Vs.v_s_x - Vs.v_s_x0)

	if Verb > 0
		println(io, "v_s_x1: ", Vs.v_s_x1)
	end

	Vs.v_s_x1 = max(Vs.v_s_x1, eps)

	if Verb > 0
		println(io, "v_s_x1: ", Vs.v_s_x1)
	end

	Vs.HD_1x = Vs.D_s_x / (C_x * Vs.Hv_s_x) - Vs.HD_2x
	Vs.HF_1x = Vs.F_s_x / Vs.Hv_s_x^(2) -
		Vs.D_s_x^(2) / (C_x * Vs.Hv_s_x^(2)) - Vs.HF_2x

	if Verb > 0
		println(io, "HD_1x: ", Vs.HD_1x)
		println(io, "HF_1x: ", Vs.HF_1x)
	end

	if Vs.HD_1x < 0.
		Vs.HD_1x = eps
	end

	if Vs.HF_1x < 0.
		Vs.HF_1x = eps
	end

	if Verb > 0
		println(io, "HD_1x: ", Vs.HD_1x)
		println(io, "HF_1x: ", Vs.HF_1x)
	end

	In_Res = I_SE_1RSB(Vs.HD_1x, Vs.HF_1x, Vs.v_s_x1, Vs.v_s_x0, L, Ax_T, Cx_T, Ax_F, Cx_F, eps)
	Vs.D_p_x = In_Res.DX
	Vs.F_p_x = In_Res.FX
	Vs.Hv_p_x1 = In_Res.HX
	Vs.Hv_p_x0 = In_Res.ChiX

	if Verb > 0
		println(io, "D_p_x: ", Vs.D_p_x)
		println(io, "F_p_x: ", Vs.F_p_x)
		println(io, "Hv_p_x1: ", Vs.Hv_p_x1)
		println(io, "Hv_p_x0: ", Vs.Hv_p_x0)
	end

	Vs.Hv_p_x1 = max(Vs.Hv_p_x1, eps)
	Vs.Hv_p_x0 = max(Vs.Hv_p_x0, eps)

	Vs.Hv_p_x = Vs.Hv_p_x0 + L * Vs.Hv_p_x1

	Mse = (C_x + Vs.F_p_x - 2 * Vs.D_p_x) / C_x
	Eps = abs(Mse - Vs.Mse_O)^(2) / Mse^(2)
	Rho = Vs.D_p_x / sqrt(Vs.F_p_x * C_x)

	A_D = Vs.D_p_x
	A_F = Vs.F_p_x
	A_H = Vs.Hv_p_x1
	A_Chi = Vs.Hv_p_x0

	Idx_a = isinf(Mse) + isnan.(Mse) +
		isinf(Eps) + isnan.(Eps) +
		isinf(Rho) + isnan.(Rho) +
		isinf(A_D) + isnan.(A_D) +
		isinf(A_F) + isnan.(A_F) +
		isinf(A_H) + isnan.(A_H) +
		isinf(A_Chi) + isnan.(A_Chi)

	if Idx_a > 0
		if Verb > 0
			println(io, "X-break: ", Idx_a)
		end
		return Vs.Mse_O, Vs.Eps_O, Vs.Rho_O, Vs.A_D_O, Vs.A_F_O, Vs.A_H_O, Vs.A_Chi_O
	end

	Vs.Mse_O = Mse
	Vs.Eps_O = Eps
	Vs.Rho_O = Rho
	Vs.A_D_O = A_D
	Vs.A_F_O = A_F
	Vs.A_H_O = A_H
	Vs.A_Chi_O = A_Chi

	if Verb > 0
		println(io, "Hv_p_x: ", Vs.Hv_p_x)
		println(io, "Mse: ", Mse, ", Eps: ", Eps, ", Rho: ", Rho,
				", A_D: ", A_D, ", A_F: ", A_F, ", A_H: ", A_H, ", A_Chi: ", A_Chi
			)
	end

	println("VASP-SE - Iter: ", t, ", Mse: ", Mse, ", Eps: ", Eps, ", Rho: ", Rho,
			", A_D: ", A_D, ", A_F: ", A_F, ", A_H: ", A_H, ", A_Chi: ", A_Chi
		)

	Vs.v_p_x0_inv = (Vs.v_s_x0 - Vs.Hv_p_x0) / (Vs.v_s_x0 * Vs.Hv_p_x0)
	Vs.v_p_x_inv = (Vs.v_s_x - Vs.Hv_p_x) / (Vs.v_s_x * Vs.Hv_p_x)

	if Verb > 0
		println(io, "v_p_x0_inv: ", Vs.v_p_x0_inv)
		println(io, "v_p_x_inv: ", Vs.v_p_x_inv)
	end

	if Vs.v_p_x0_inv < eps
		if Verb > 0
			println(io, "Res_v_p_x0_inv: !!!")
		end

		Vs.v_p_x0_inv = Vs.v_p_x0_inv_O
	end

	if Vs.v_p_x_inv < eps
		if Verb > 0
			println(io, "Res_v_p_x_inv: !!!")
		end

		Vs.v_p_x_inv = Vs.v_p_x_inv_O
	end

	if Verb > 0
		println(io, "v_p_x0_inv: ", Vs.v_p_x0_inv)
		println(io, "v_p_x_inv: ", Vs.v_p_x_inv)
	end

	Vs.v_p_x0_inv = Mes * Vs.v_p_x0_inv + (1 - Mes) * Vs.v_p_x0_inv_O
	Vs.v_p_x0_inv_O = Vs.v_p_x0_inv
	Vs.v_p_x_inv = Mes * Vs.v_p_x_inv + (1 - Mes) * Vs.v_p_x_inv_O
	Vs.v_p_x_inv_O = Vs.v_p_x_inv

	if Verb > 0
		println(io, "v_p_x0_inv: ", Vs.v_p_x0_inv)
		println(io, "v_p_x_inv: ", Vs.v_p_x_inv)
	end

	T_v_p_x0 = 1 / Vs.v_p_x0_inv
	T_v_p_x = 1 / Vs.v_p_x_inv

	if Verb > 0
		println(io, "T_v_p_x0: ", T_v_p_x0)
		println(io, "T_v_p_x: ", T_v_p_x)
	end

	if Verb > 0
		tmp_T_v_p_x0 = T_v_p_x0 + eps
		Res_T_v_p_x0 = (T_v_p_x >= tmp_T_v_p_x0)
		println(io, "Res_T_v_p_x0: ", Res_T_v_p_x0)
	end

	T_v_p_x = max(T_v_p_x, T_v_p_x0 + eps)

	if Verb > 0
		println(io, "T_v_p_x: ", T_v_p_x)
	end

	Vs.v_p_x_inv = 1 / T_v_p_x
	Vs.v_p_x_inv_O = Vs.v_p_x_inv

	if Verb > 0
		println(io, "v_p_x0_inv: ", Vs.v_p_x0_inv)
		println(io, "v_p_x_inv: ", Vs.v_p_x_inv)
	end

	Vs.HD_2x = Vs.D_p_x / (C_x * Vs.Hv_p_x) - Vs.HD_1x
	Vs.HF_2x = Vs.F_p_x / Vs.Hv_p_x^(2) -
		Vs.D_p_x^(2) / (C_x * Vs.Hv_p_x^(2)) - Vs.HF_1x

	if Verb > 0
		println(io, "HD_2x: ", Vs.HD_2x)
		println(io, "HF_2x: ", Vs.HF_2x)
	end

	if Vs.HD_2x < 0.
		Vs.HD_2x = eps
	end

	if Vs.HF_2x < 0.
		Vs.HF_2x = eps
	end

	if Verb > 0
		println(io, "HD_2x: ", Vs.HD_2x)
		println(io, "HF_2x: ", Vs.HF_2x)
	end

	Vs.Hv_p_z0 = 1 / M * sum(
			HHt.S ./ (
				Vs.v_p_x0_inv .+ Vs.v_s_z0_inv * HHt.S
			)
		)
	Vs.Hv_p_z = 1 / M * sum(
			HHt.S ./ (
				Vs.v_p_x_inv .+ Vs.v_s_z_inv * HHt.S
			)
		)

	if Verb > 0
		println(io, "Hv_p_z0: ", Vs.Hv_p_z0)
		println(io, "Hv_p_z: ", Vs.Hv_p_z)
	end

	Vs.Hv_p_z0 = max(Vs.Hv_p_z0, eps)
	Vs.Hv_p_z = max(Vs.Hv_p_z, eps)

	if Verb > 0
		println(io, "Hv_p_z0: ", Vs.Hv_p_z0)
		println(io, "Hv_p_z: ", Vs.Hv_p_z)
	end

	if Verb > 0
		tmp_Hv_p_z0 = Vs.Hv_p_z0 + eps
		Res_Hv_p_z0 = (Vs.Hv_p_z >= tmp_Hv_p_z0)
		println(io, "Res_Hv_p_z0: ", Res_Hv_p_z0)
	end

	Vs.Hv_p_z = max(Vs.Hv_p_z, Vs.Hv_p_z0 + eps)

	if Verb > 0
		println(io, "Hv_p_z: ", Vs.Hv_p_z)
	end

	Vs.D_p_z = C_x / M * sum(
			HHt.S .* (
				Vs.HD_2x .+ Vs.HD_2z * HHt.S
			) ./ (
				Vs.v_p_x_inv .+ Vs.v_s_z_inv * HHt.S
			)
		)
	Vs.F_p_z = C_x / M * sum(
		HHt.S .* (
			Vs.HD_2x .+ Vs.HD_2z * HHt.S
		).^(2) ./ (
			Vs.v_p_x_inv .+ Vs.v_s_z_inv * HHt.S
		).^(2)
	) + 1 / M * sum(
		HHt.S .* (
			Vs.HF_2x .+ Vs.HF_2z * HHt.S
		) ./ (
			Vs.v_p_x_inv .+ Vs.v_s_z_inv * HHt.S
		).^(2)
	)

	if Verb > 0
		println(io, "D_p_z: ", Vs.D_p_z)
		println(io, "F_p_z: ", Vs.F_p_z)
	end

	Vs.v_p_z0_inv = 1 / Vs.Hv_p_z0 - Vs.v_s_z0_inv
	Vs.v_p_z_inv = 1 / Vs.Hv_p_z - Vs.v_s_z_inv

	if Verb > 0
		println(io, "v_p_z0_inv: ", Vs.v_p_z0_inv)
		println(io, "v_p_z_inv: ", Vs.v_p_z_inv)
	end

	if Vs.v_p_z0_inv < 0
		if Verb > 0
			println(io, "Res_v_p_z0_inv: !!!")
		end

		Vs.v_p_z0_inv = Vs.v_p_z0_inv_O
	end

	if Vs.v_p_z_inv < 0
		if Verb > 0
			println(io, "Res_v_p_z_inv: !!!")
		end

		Vs.v_p_z_inv = Vs.v_p_z_inv_O
	end

	if Verb > 0
		println(io, "v_p_z0_inv: ", Vs.v_p_z0_inv)
		println(io, "v_p_z_inv: ", Vs.v_p_z_inv)
	end

	Vs.v_p_z0_inv = Mes * Vs.v_p_z0_inv + (1 - Mes) * Vs.v_p_z0_inv_O
	Vs.v_p_z0_inv_O = Vs.v_p_z0_inv
	Vs.v_p_z_inv = Mes * Vs.v_p_z_inv + (1 - Mes) * Vs.v_p_z_inv_O
	Vs.v_p_z_inv_O = Vs.v_p_z_inv

	if Verb > 0
		println(io, "v_p_z0_inv: ", Vs.v_p_z0_inv)
		println(io, "v_p_z_inv: ", Vs.v_p_z_inv)
	end

	T_v_p_z0 = 1 / Vs.v_p_z0_inv
	T_v_p_z = 1 / Vs.v_p_z_inv

	if Verb > 0
		println(io, "T_v_p_z0: ", T_v_p_z0)
		println(io, "T_v_p_z: ", T_v_p_z)
	end

	if Verb > 0
		tmp_T_v_p_z0 = T_v_p_z0 + eps
		Res_T_v_p_z0 = (T_v_p_z >= tmp_T_v_p_z0)
		println(io, "Res_T_v_p_z0: ", Res_T_v_p_z0)
	end

	T_v_p_z = max(T_v_p_z, T_v_p_z0 + eps)

	if Verb > 0
		println(io, "T_v_p_z: ", T_v_p_z)
	end

	Vs.v_p_z_inv = 1 / T_v_p_z
	Vs.v_p_z_inv_O = Vs.v_p_z_inv

	if Verb > 0
		println(io, "v_p_z0_inv: ", Vs.v_p_z0_inv)
		println(io, "v_p_z_inv: ", Vs.v_p_z_inv)
	end

	Vs.HD_1z = Vs.D_p_z / (C_z * Vs.Hv_p_z) - Vs.HD_2z
	Vs.HF_1z = Vs.F_p_z / Vs.Hv_p_z^(2) -
		Vs.D_p_z^(2) / (C_z * Vs.Hv_p_z^(2)) - Vs.HF_2z

	if Verb > 0
		println(io, "HD_1z: ", Vs.HD_1z)
		println(io, "HF_1z: ", Vs.HF_1z)
	end

	if Vs.HD_1z < 0.
		Vs.HD_1z = eps
	end

	if Vs.HF_1z < 0.
		Vs.HF_1z = eps
	end

	if Verb > 0
		println(io, "HD_1z: ", Vs.HD_1z)
		println(io, "HF_1z: ", Vs.HF_1z)
	end

	return Mse, Eps, Rho, A_D, A_F, A_H, A_Chi
end

Solve(Prob; kws ...) = SE(Prob, Str_1RSB_Params(; kws ...))

function SE(Prob, Prms::Str_1RSB_Params)
	@extract Prob: N M H x0 y
	@extract Prms: Epochs Sd Epsilon Verb Name

	Sd > 0 && Random.seed!(Sd)

	### Print utilities ###
	Df = DataFrame(
			Epoch = Int[], Mse = Float64[], Eps = Float64[], Rho = Float64[],
			A_D = Float64[], A_F = Float64[], A_H = Float64[], A_Chi = Float64[]
		)

	Vs = Setting(Prob)
	Init!(Vs, Prob, Prms)

	Ok = false

	io = 0

	if Verb > 0
		rm("./$Name-S_VASP_SE.txt", force = true)
		io = open("./$Name-S_VASP_SE.txt", "w")

		rm("./$Name-S_Vs_Df.bson", force = true)
		rm("./$Name-S_Vs.bson", force = true)
		rm("./$Name-S_Vs_Prms.bson", force = true)
	end

	Mse_O = 1e3
	Eps_O = 1e3
	Rho_O = 1e3

	A_D_O = 1e3
	A_F_O = 1e3
	A_H_O = 1e3
	A_Chi_O = 1e3

	for i = 1 : Epochs
		Mse, Eps, Rho, A_D, A_F, A_H, A_Chi = Iter!(Vs, i, Prob, Prms, io, x0)

		# Ok = Mse_O < Mse

		# if Ok
		# 	break
		# end

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
		S_Vs_Df = deepcopy(Df)
		S_Vs = deepcopy(Vs)
		S_Vs_Prms = deepcopy(Prms)

		BSON.@save "./$Name-S_Vs_Df.bson" S_Vs_Df
		BSON.@save "./$Name-S_Vs.bson" S_Vs
		BSON.@save "./$Name-S_Vs_Prms.bson" S_Vs_Prms

		close(io)
	end

	!Ok && @warn("Not converged!")
	return Df, Vs, Ok, Prms
end

end	# Module