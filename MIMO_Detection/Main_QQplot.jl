# run code in a terminal of Ubuntu -> export JULIA_NUM_THREADS=4
include("./Common.jl")
include("./GVASP_QQplot.jl")
include("./GVASP_SE_QQplot.jl")

### Set the parameters of GLE ###
Prior = :BPSK

N = 5000
# N = 4000
# N = 2000
# N = 1024
# N = 1000

# a = 2.00
# a = 1.50
# a = 1.20
# a = 1.10
a = 1.00
# a = 0.95
# a = 0.90
# a = 0.80

Seed = 0
# Seed = 200

# Ax_T = abs(2)
# Ax_T = abs(1.5)
# Ax_T = abs(1.2)
Ax_T = abs(1.0)
# Ax_T = abs(1 / sqrt(N))

# vw_T = 0.20
# vw_T = 0.15
# vw_T = 0.12
vw_T = 0.10
# vw_T = 0.05

# cor_flag = false
# cor = 0.

cor_flag = true
# cor = 0.4
# cor = 0.35
# cor = 0.3
cor = 0.25
# cor = 0.2

### Set the parameters of GAMP, GVAMP, GASP and GVASP ###
Epochs = 30

# L = 100
L = 10

Sd = Seed

InitX = :Spectral
# InitX = nothing

Epsilon = 1e-3

Mes = 1.0
# Mes = 0.95
# Mes = 0.90
# Mes = 0.85

Ax_F = Ax_T

# vw_F = 0.50
# vw_F = 0.40
# vw_F = 0.30
# vw_F = 0.20
vw_F = 0.15
# vw_F = vw_T

EPS = 2e-16

# Verb = 1
Verb = 0

Num = 1

M = round(Int, N * a)

Gvasp_Mse_List = zeros(Epochs, Num)
Gvasp_Eps_List = zeros(Epochs, Num)
Gvasp_Rho_List = zeros(Epochs, Num)
Gvasp_A_D_List = zeros(Epochs, Num)
Gvasp_A_F_List = zeros(Epochs, Num)
Gvasp_A_H_List = zeros(Epochs, Num)
Gvasp_A_GsF_List = zeros(Epochs, Num)

Gvasp_S_v_s_z_List = zeros(Epochs, Num)
Gvasp_S_HD_2z_List = zeros(Epochs, Num)
Gvasp_S_HF_2z_List = zeros(Epochs, Num)

Gvasp_S_v_s_x_List = zeros(Epochs, Num)
Gvasp_S_HD_1x_List = zeros(Epochs, Num)
Gvasp_S_HF_1x_List = zeros(Epochs, Num)

Gvasp_S_v_p_x_List = zeros(Epochs, Num)
Gvasp_S_HD_2x_List = zeros(Epochs, Num)
Gvasp_S_HF_2x_List = zeros(Epochs, Num)

Gvasp_S_v_p_z_List = zeros(Epochs, Num)
Gvasp_S_HD_1z_List = zeros(Epochs, Num)
Gvasp_S_HF_1z_List = zeros(Epochs, Num)

Gvasp_S_m_s_z_Df_List = zeros(M, Epochs, Num)
Gvasp_S_m_s_x_Df_List = zeros(N, Epochs, Num)
Gvasp_S_m_p_x_Df_List = zeros(N, Epochs, Num)
Gvasp_S_m_p_z_Df_List = zeros(M, Epochs, Num)

Gvasp_SE_Mse_List = zeros(Epochs, Num)
Gvasp_SE_Eps_List = zeros(Epochs, Num)
Gvasp_SE_Rho_List = zeros(Epochs, Num)
Gvasp_SE_A_D_List = zeros(Epochs, Num)
Gvasp_SE_A_F_List = zeros(Epochs, Num)
Gvasp_SE_A_H_List = zeros(Epochs, Num)
Gvasp_SE_A_GsF_List = zeros(Epochs, Num)

Gvasp_SE_S_v_s_z_List = zeros(Epochs, Num)
Gvasp_SE_S_HD_2z_List = zeros(Epochs, Num)
Gvasp_SE_S_HF_2z_List = zeros(Epochs, Num)

Gvasp_SE_S_v_s_x_List = zeros(Epochs, Num)
Gvasp_SE_S_HD_1x_List = zeros(Epochs, Num)
Gvasp_SE_S_HF_1x_List = zeros(Epochs, Num)

Gvasp_SE_S_v_p_x_List = zeros(Epochs, Num)
Gvasp_SE_S_HD_2x_List = zeros(Epochs, Num)
Gvasp_SE_S_HF_2x_List = zeros(Epochs, Num)

Gvasp_SE_S_v_p_z_List = zeros(Epochs, Num)
Gvasp_SE_S_HD_1z_List = zeros(Epochs, Num)
Gvasp_SE_S_HF_1z_List = zeros(Epochs, Num)

x0_List = zeros(N, Num)
z_List = zeros(M, Num)

U_List = zeros(M, M, Num)
V_List = zeros(N, N, Num)

Int_cor_flag = Int(cor_flag)

Name = "Main_QQplot-cf$Int_cor_flag-c$cor-N$N-a$a-L$L-aT$Ax_T-vT$vw_T-aF$Ax_F-vF$vw_F"

rm("./$Name-Gvasp_Df_List.bson", force = true)

rm("./$Name-Gvasp_S_m_s_z_Df_List", force = true)
rm("./$Name-Gvasp_S_m_s_x_Df_List", force = true)
rm("./$Name-Gvasp_S_m_p_x_Df_List", force = true)
rm("./$Name-Gvasp_S_m_p_z_Df_List", force = true)

rm("./$Name-Gvasp_SE_Df_List.bson", force = true)

rm("./$Name-x0_List", force = true)
rm("./$Name-z_List", force = true)

rm("./$Name-U_List", force = true)
rm("./$Name-V_List", force = true)

d1 = DateTime(now())

println("nthreads = ", Threads.nthreads())

Threads.@threads for i = 1 : Num
	println("")
	println("i: ", i)

	### Define the problem ###
	Prob = Problem_Setting(
			"GLE"; Prior = Prior,
			N = N, a = a, Seed = Seed,
			Ax_T = Ax_T,
			vw_T = vw_T,
			cor_flag = cor_flag,
			cor = cor
		)

	# ### Solve the problem by GVASP ###

	println("GVASP")

	Gvasp_Df, Gvasp_S_m_s_z_Df, Gvasp_S_m_s_x_Df, Gvasp_S_m_p_x_Df, Gvasp_S_m_p_z_Df,
	Gvasp, Gvasp_Ok, Gvasp_Prms = Module_GVASP_QQplot.Solve(Prob,
			Epochs = Epochs,
			L = L,
			Sd = Sd,
			InitX = InitX,
			Epsilon = Epsilon,
			Mes = Mes,
			Ax_F = Ax_F,
			vw_F = vw_F,
			EPS = EPS,
			Verb = Verb,
			Name = Name
		)

	Gvasp_Mse_List[:, i] .= Gvasp_Df.Mse
	Gvasp_Eps_List[:, i] .= Gvasp_Df.Eps
	Gvasp_Rho_List[:, i] .= Gvasp_Df.Rho
	Gvasp_A_D_List[:, i] .= Gvasp_Df.A_D
	Gvasp_A_F_List[:, i] .= Gvasp_Df.A_F
	Gvasp_A_H_List[:, i] .= Gvasp_Df.A_H
	Gvasp_A_GsF_List[:, i] .= Gvasp_Df.A_GsF

	Gvasp_S_v_s_z_List[:, i] .= Gvasp_Df.S_v_s_z
	Gvasp_S_HD_2z_List[:, i] .= Gvasp_Df.S_HD_2z
	Gvasp_S_HF_2z_List[:, i] .= Gvasp_Df.S_HF_2z
	
	Gvasp_S_v_s_x_List[:, i] .= Gvasp_Df.S_v_s_x
	Gvasp_S_HD_1x_List[:, i] .= Gvasp_Df.S_HD_1x
	Gvasp_S_HF_1x_List[:, i] .= Gvasp_Df.S_HF_1x
	
	Gvasp_S_v_p_x_List[:, i] .= Gvasp_Df.S_v_p_x
	Gvasp_S_HD_2x_List[:, i] .= Gvasp_Df.S_HD_2x
	Gvasp_S_HF_2x_List[:, i] .= Gvasp_Df.S_HF_2x
	
	Gvasp_S_v_p_z_List[:, i] .= Gvasp_Df.S_v_p_z
	Gvasp_S_HD_1z_List[:, i] .= Gvasp_Df.S_HD_1z
	Gvasp_S_HF_1z_List[:, i] .= Gvasp_Df.S_HF_1z

	Gvasp_S_m_s_z_Df_List[:, :, i] .= Gvasp_S_m_s_z_Df
	Gvasp_S_m_s_x_Df_List[:, :, i] .= Gvasp_S_m_s_x_Df
	Gvasp_S_m_p_x_Df_List[:, :, i] .= Gvasp_S_m_p_x_Df
	Gvasp_S_m_p_z_Df_List[:, :, i] .= Gvasp_S_m_p_z_Df

	x0_List[:, i] .= Prob.x0
	z_List[:, i] .= Prob.z

	U_List[:, :, i] .= Gvasp.U
	V_List[:, :, i] .= Gvasp.V

	println("GVASP-SE")
	Gvasp_SE_Df, Gvasp_SE, Gvasp_SE_Ok, Gvasp_SE_Prms = Module_GVASP_SE_QQplot.Solve(Prob,
			Epochs = Epochs,
			L = L,
			Sd = Sd,
			InitX = InitX,
			Epsilon = Epsilon,
			Mes = Mes,
			Ax_F = Ax_F,
			vw_F = vw_F,
			EPS = EPS,
			Verb = Verb,
			Name = Name
		)

	Gvasp_SE_Mse_List[:, i] .= Gvasp_SE_Df.Mse
	Gvasp_SE_Eps_List[:, i] .= Gvasp_SE_Df.Eps
	Gvasp_SE_Rho_List[:, i] .= Gvasp_SE_Df.Rho
	Gvasp_SE_A_D_List[:, i] .= Gvasp_SE_Df.A_D
	Gvasp_SE_A_F_List[:, i] .= Gvasp_SE_Df.A_F
	Gvasp_SE_A_H_List[:, i] .= Gvasp_SE_Df.A_H
	Gvasp_SE_A_GsF_List[:, i] .= Gvasp_SE_Df.A_GsF

	Gvasp_SE_S_v_s_z_List[:, i] .= Gvasp_SE_Df.S_v_s_z
	Gvasp_SE_S_HD_2z_List[:, i] .= Gvasp_SE_Df.S_HD_2z
	Gvasp_SE_S_HF_2z_List[:, i] .= Gvasp_SE_Df.S_HF_2z
	
	Gvasp_SE_S_v_s_x_List[:, i] .= Gvasp_SE_Df.S_v_s_x
	Gvasp_SE_S_HD_1x_List[:, i] .= Gvasp_SE_Df.S_HD_1x
	Gvasp_SE_S_HF_1x_List[:, i] .= Gvasp_SE_Df.S_HF_1x
	
	Gvasp_SE_S_v_p_x_List[:, i] .= Gvasp_SE_Df.S_v_p_x
	Gvasp_SE_S_HD_2x_List[:, i] .= Gvasp_SE_Df.S_HD_2x
	Gvasp_SE_S_HF_2x_List[:, i] .= Gvasp_SE_Df.S_HF_2x
	
	Gvasp_SE_S_v_p_z_List[:, i] .= Gvasp_SE_Df.S_v_p_z
	Gvasp_SE_S_HD_1z_List[:, i] .= Gvasp_SE_Df.S_HD_1z
	Gvasp_SE_S_HF_1z_List[:, i] .= Gvasp_SE_Df.S_HF_1z

end

Gvasp_Df_List = Df_Push_New(
		Epochs, Gvasp_Mse_List, Gvasp_Eps_List,
		Gvasp_Rho_List,
		Gvasp_A_D_List, Gvasp_A_F_List, Gvasp_A_H_List, Gvasp_A_GsF_List,
		Gvasp_S_v_s_z_List, Gvasp_S_HD_2z_List, Gvasp_S_HF_2z_List,
		Gvasp_S_v_s_x_List, Gvasp_S_HD_1x_List, Gvasp_S_HF_1x_List,
		Gvasp_S_v_p_x_List, Gvasp_S_HD_2x_List, Gvasp_S_HF_2x_List,
		Gvasp_S_v_p_z_List, Gvasp_S_HD_1z_List, Gvasp_S_HF_1z_List
	)
println("-----  GVASP_QQplot  -----")
println(Gvasp_Df_List)

BSON.@save "./$Name-Gvasp_Df_List.bson" Gvasp_Df_List

BSON.@save "./$Name-Gvasp_S_m_s_z_Df_List" Gvasp_S_m_s_z_Df_List
BSON.@save "./$Name-Gvasp_S_m_s_x_Df_List" Gvasp_S_m_s_x_Df_List
BSON.@save "./$Name-Gvasp_S_m_p_x_Df_List" Gvasp_S_m_p_x_Df_List
BSON.@save "./$Name-Gvasp_S_m_p_z_Df_List" Gvasp_S_m_p_z_Df_List

Gvasp_SE_Df_List = Df_Push_New(
		Epochs, Gvasp_SE_Mse_List, Gvasp_SE_Eps_List,
		Gvasp_SE_Rho_List,
		Gvasp_SE_A_D_List, Gvasp_SE_A_F_List, Gvasp_SE_A_H_List, Gvasp_SE_A_GsF_List,
		Gvasp_SE_S_v_s_z_List, Gvasp_SE_S_HD_2z_List, Gvasp_SE_S_HF_2z_List,
		Gvasp_SE_S_v_s_x_List, Gvasp_SE_S_HD_1x_List, Gvasp_SE_S_HF_1x_List,
		Gvasp_SE_S_v_p_x_List, Gvasp_SE_S_HD_2x_List, Gvasp_SE_S_HF_2x_List,
		Gvasp_SE_S_v_p_z_List, Gvasp_SE_S_HD_1z_List, Gvasp_SE_S_HF_1z_List
	)
println("-----  GVASP-SE  -----")
println(Gvasp_SE_Df_List)

BSON.@save "./$Name-Gvasp_SE_Df_List.bson" Gvasp_SE_Df_List

BSON.@save "./$Name-x0_List" x0_List
BSON.@save "./$Name-z_List" z_List

BSON.@save "./$Name-U_List" U_List
BSON.@save "./$Name-V_List" V_List

d2 = DateTime(now())
println("Times: ", d2 - d1)