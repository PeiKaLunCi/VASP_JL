# run code in a terminal of Ubuntu -> export JULIA_NUM_THREADS=4
include("./Common.jl")
include("./GASP.jl")
include("./GVASP.jl")
include("./GVASP_SE.jl")

### Set the parameters of GLE ###
Prior = :Gauss

N = 1000

a = 8.0

Seed = 0

C_x = 1.
Mask = 1.
vw = 0.
Act = abs

cor_flag = false
cor = 0.

# cor_flag = true
# cor = 0.4

### Set the parameters of GASP and GVASP ###
Epochs = 30

L = 10
# L = 100

# La = 1e-3
La = 1e-5
# La = 0
Drop_La = false

Sd = Seed

InitX = :Spectral

Ro = 1e-3

Epsilon = 1e-3

Mes = 1.0

va = 1 / 2
# va = 1.

EPS = 2e-16

# Verb = 1
Verb = 0

Num = 40

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

Name = "Main_Iter-cf$Int_cor_flag-c$cor-N$N-a$a-L$L-va$va"

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
	println("i: ", i)

	### Define the problem ###
	Prob = Problem_Setting(
			"GLE"; Prior = Prior,
			N = N, a = a, Seed = Seed,
			C_x = C_x,
			Mask = Mask,
			vw = vw,
			Act = Act,
			cor_flag = cor_flag,
			cor = cor
		)

	# ### Solve the problem by GASP and GVASP ###
	println("GASP")
	Gasp_Df, Gasp, Gasp_Ok, Gasp_Prms = Module_GASP.Solve(Prob,
			Epochs = Epochs,
			L = L,
			La = La,
			Drop_La = Drop_La,
			Sd = Sd,
			InitX = InitX,
			Ro = Ro,
			Epsilon = Epsilon,
			Mes = Mes,
			va = va,
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
			La = La,
			Drop_La = Drop_La,
			Sd = Sd,
			InitX = InitX,
			Ro = Ro,
			Epsilon = Epsilon,
			Mes = Mes,
			va = va,
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
			La = La,
			Drop_La = Drop_La,
			Sd = Sd,
			InitX = InitX,
			Ro = Ro,
			Epsilon = Epsilon,
			Mes = Mes,
			va = va,
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

Mse_List = zeros(Epochs, 3)
Mse_List[:, 1] .= 10 * log10.(Gasp_Df_Final.Mse)
Mse_List[:, 2] .= 10 * log10.(Gvasp_Df_Final.Mse)
Mse_List[:, 3] .= 10 * log10.(Gvasp_SE_Df_Final.Mse)

fig_MSE = plot(
		Epochs_List, Mse_List,
		title = "GASP-GVASP",
		xlabel = "Iter", ylabel = "MSE",
		label = ["GASP" "GVASP" "GVASP-SE"]
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