# run code in a terminal of Ubuntu -> export JULIA_NUM_THREADS=4
include("./Common.jl")
include("./GAMP.jl")
include("./GVAMP.jl")
include("./GASP.jl")
include("./GVASP.jl")
include("./GVASP_SE.jl")

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

# Num = 400
Num = 200
# Num = 10

Gamp_Mse_List = zeros(Epochs, Num)
Gamp_Eps_List = zeros(Epochs, Num)
Gamp_Rho_List = zeros(Epochs, Num)
Gamp_A_D_List = zeros(Epochs, Num)
Gamp_A_F_List = zeros(Epochs, Num)
Gamp_A_H_List = zeros(Epochs, Num)
Gamp_A_GsF_List = zeros(Epochs, Num)

Gvamp_Mse_List = zeros(Epochs, Num)
Gvamp_Eps_List = zeros(Epochs, Num)
Gvamp_Rho_List = zeros(Epochs, Num)
Gvamp_A_D_List = zeros(Epochs, Num)
Gvamp_A_F_List = zeros(Epochs, Num)
Gvamp_A_H_List = zeros(Epochs, Num)
Gvamp_A_GsF_List = zeros(Epochs, Num)

Gasp_Mse_List = zeros(Epochs, Num)
Gasp_Eps_List = zeros(Epochs, Num)
Gasp_Rho_List = zeros(Epochs, Num)
Gasp_A_D_List = zeros(Epochs, Num)
Gasp_A_F_List = zeros(Epochs, Num)
Gasp_A_H_List = zeros(Epochs, Num)
Gasp_A_GsF_List = zeros(Epochs, Num)

Gvasp_Mse_List = zeros(Epochs, Num)
Gvasp_Eps_List = zeros(Epochs, Num)
Gvasp_Rho_List = zeros(Epochs, Num)
Gvasp_A_D_List = zeros(Epochs, Num)
Gvasp_A_F_List = zeros(Epochs, Num)
Gvasp_A_H_List = zeros(Epochs, Num)
Gvasp_A_GsF_List = zeros(Epochs, Num)

Gvasp_SE_Mse_List = zeros(Epochs, Num)
Gvasp_SE_Eps_List = zeros(Epochs, Num)
Gvasp_SE_Rho_List = zeros(Epochs, Num)
Gvasp_SE_A_D_List = zeros(Epochs, Num)
Gvasp_SE_A_F_List = zeros(Epochs, Num)
Gvasp_SE_A_H_List = zeros(Epochs, Num)
Gvasp_SE_A_GsF_List = zeros(Epochs, Num)

Int_cor_flag = Int(cor_flag)

Name = "Main_Iter-cf$Int_cor_flag-c$cor-N$N-a$a-L$L-aT$Ax_T-vT$vw_T-aF$Ax_F-vF$vw_F"

rm("./$Name-S_Gamp_Df_Final.bson", force = true)
rm("./$Name-S_Gvamp_Df_Final.bson", force = true)
rm("./$Name-S_Gasp_Df_Final.bson", force = true)
rm("./$Name-S_Gvasp_Df_Final.bson", force = true)
rm("./$Name-S_Gvasp_SE_Df_Final.bson", force = true)

rm("./$Name-fig_MSE.pdf", force = true)
rm("./$Name-fig_D.pdf", force = true)
rm("./$Name-fig_F.pdf", force = true)
rm("./$Name-fig_H.pdf", force = true)
rm("./$Name-fig_GsF.pdf", force = true)

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

	# ### Solve the problem by GAMP, GVAMP, GASP and GVASP ###
	println("GAMP")
	Gamp_Df, Gamp, Gamp_Ok, Gamp_Prms = Module_GAMP.Solve(Prob,
			Epochs = Epochs,
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

	Gamp_Mse_List[:, i] .= Gamp_Df.Mse
	Gamp_Eps_List[:, i] .= Gamp_Df.Eps
	Gamp_Rho_List[:, i] .= Gamp_Df.Rho
	Gamp_A_D_List[:, i] .= Gamp_Df.A_D
	Gamp_A_F_List[:, i] .= Gamp_Df.A_F
	Gamp_A_H_List[:, i] .= Gamp_Df.A_H
	Gamp_A_GsF_List[:, i] .= Gamp_Df.A_GsF

	println("GVAMP")
	Gvamp_Df, Gvamp, Gvamp_Ok, Gvamp_Prms = Module_GVAMP.Solve(Prob,
			Epochs = Epochs,
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

	Gvamp_Mse_List[:, i] .= Gvamp_Df.Mse
	Gvamp_Eps_List[:, i] .= Gvamp_Df.Eps
	Gvamp_Rho_List[:, i] .= Gvamp_Df.Rho
	Gvamp_A_D_List[:, i] .= Gvamp_Df.A_D
	Gvamp_A_F_List[:, i] .= Gvamp_Df.A_F
	Gvamp_A_H_List[:, i] .= Gvamp_Df.A_H
	Gvamp_A_GsF_List[:, i] .= Gvamp_Df.A_GsF

	println("GASP")
	Gasp_Df, Gasp, Gasp_Ok, Gasp_Prms = Module_GASP.Solve(Prob,
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

	Gasp_Mse_List[:, i] .= Gasp_Df.Mse
	Gasp_Eps_List[:, i] .= Gasp_Df.Eps
	Gasp_Rho_List[:, i] .= Gasp_Df.Rho
	Gasp_A_D_List[:, i] .= Gasp_Df.A_D
	Gasp_A_F_List[:, i] .= Gasp_Df.A_F
	Gasp_A_H_List[:, i] .= Gasp_Df.A_H
	Gasp_A_GsF_List[:, i] .= Gasp_Df.A_GsF

	println("GVASP")
	Gvasp_Df, Gvasp, Gvasp_Ok, Gvasp_Prms = Module_GVASP.Solve(Prob,
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

	println("GVASP-SE")
	Gvasp_SE_Df, Gvasp_SE, Gvasp_SE_Ok, Gvasp_SE_Prms = Module_GVASP_SE.Solve(Prob,
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
end

Gamp_Df_Final = Df_Push(
		Epochs, Gamp_Mse_List, Gamp_Eps_List,
		Gamp_Rho_List,
		Gamp_A_D_List, Gamp_A_F_List, Gamp_A_H_List, Gamp_A_GsF_List
	)
println("-----  GAMP  -----")
println(Gamp_Df_Final)
BSON.@save "./$Name-S_Gamp_Df_Final.bson" Gamp_Df_Final

Gvamp_Df_Final = Df_Push(
		Epochs, Gvamp_Mse_List, Gvamp_Eps_List,
		Gvamp_Rho_List,
		Gvamp_A_D_List, Gvamp_A_F_List, Gvamp_A_H_List, Gvamp_A_GsF_List
	)
println("-----  GVAMP  -----")
println(Gvamp_Df_Final)
BSON.@save "./$Name-S_Gvamp_Df_Final.bson" Gvamp_Df_Final

Gasp_Df_Final = Df_Push(
		Epochs, Gasp_Mse_List, Gasp_Eps_List,
		Gasp_Rho_List,
		Gasp_A_D_List, Gasp_A_F_List, Gasp_A_H_List, Gasp_A_GsF_List
	)
println("-----  GASP  -----")
println(Gasp_Df_Final)
BSON.@save "./$Name-S_Gasp_Df_Final.bson" Gasp_Df_Final

Gvasp_Df_Final = Df_Push(
		Epochs, Gvasp_Mse_List, Gvasp_Eps_List,
		Gvasp_Rho_List,
		Gvasp_A_D_List, Gvasp_A_F_List, Gvasp_A_H_List, Gvasp_A_GsF_List
	)
println("-----  GVASP  -----")
println(Gvasp_Df_Final)
BSON.@save "./$Name-S_Gvasp_Df_Final.bson" Gvasp_Df_Final

Gvasp_SE_Df_Final = Df_Push(
		Epochs, Gvasp_SE_Mse_List, Gvasp_SE_Eps_List,
		Gvasp_SE_Rho_List,
		Gvasp_SE_A_D_List, Gvasp_SE_A_F_List, Gvasp_SE_A_H_List, Gvasp_SE_A_GsF_List
	)
println("-----  GVASP-SE  -----")
println(Gvasp_SE_Df_Final)
BSON.@save "./$Name-S_Gvasp_SE_Df_Final.bson" Gvasp_SE_Df_Final

d2 = DateTime(now())
println("Times: ", d2 - d1)

Epochs_List = [1 : Epochs ...]

Mse_List = zeros(Epochs, 5)
Mse_List[:, 1] .= 10 * log10.(Gamp_Df_Final.Mse)
Mse_List[:, 2] .= 10 * log10.(Gvamp_Df_Final.Mse)
Mse_List[:, 3] .= 10 * log10.(Gasp_Df_Final.Mse)
Mse_List[:, 4] .= 10 * log10.(Gvasp_Df_Final.Mse)
Mse_List[:, 5] .= 10 * log10.(Gvasp_SE_Df_Final.Mse)

fig_MSE = plot(
		Epochs_List, Mse_List,
		title = "GAMP-GVAMP-GASP-GVASP",
		xlabel = "Iter", ylabel = "MSE",
		label = ["GAMP" "GVAMP" "GASP" "GVASP" "GVASP-SE"]
	)
savefig(fig_MSE, "./$Name-fig_MSE.pdf")

D_List = zeros(Epochs, 2)
D_List[:, 1] .= Gvasp_Df_Final.A_D
D_List[:, 2] .= Gvasp_SE_Df_Final.A_D
fig_D = plot(
		Epochs_List, D_List,
		title = "GVASP: Algo vs SE",
		xlabel = "Iter", ylabel = "D",
		label = ["D" "SE-D"]
	)
savefig(fig_D, "./$Name-fig_D.pdf")

F_List = zeros(Epochs, 2)
F_List[:, 1] .= Gvasp_Df_Final.A_F
F_List[:, 2] .= Gvasp_SE_Df_Final.A_F
fig_F = plot(
		Epochs_List, F_List,
		title = "GVASP: Algo vs SE",
		xlabel = "Iter", ylabel = "F",
		label = ["F" "SE-F"]
	)
savefig(fig_F, "./$Name-fig_F.pdf")

H_List = zeros(Epochs, 2)
H_List[:, 1] .= Gvasp_Df_Final.A_H
H_List[:, 2] .= Gvasp_SE_Df_Final.A_H
fig_H = plot(
		Epochs_List, H_List,
		title = "GVASP: Algo vs SE",
		xlabel = "Iter", ylabel = "H",
		label = ["H" "SE-H"]
	)
savefig(fig_H, "./$Name-fig_H.pdf")

GsF_List = zeros(Epochs, 2)
GsF_List[:, 1] .= Gvasp_Df_Final.A_GsF
GsF_List[:, 2] .= Gvasp_SE_Df_Final.A_GsF
fig_GsF = plot(
		Epochs_List, GsF_List,
		title = "GVASP: Algo vs SE",
		xlabel = "Iter", ylabel = "GsF",
		label = ["GsF" "SE-GsF"]
	)
savefig(fig_GsF, "./$Name-fig_GsF.pdf")