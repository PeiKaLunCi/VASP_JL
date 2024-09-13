# run code in a terminal of Ubuntu -> export JULIA_NUM_THREADS=4
include("./Common.jl")
include("./VAMP.jl")
include("./ASP.jl")
include("./VASP.jl")
include("./VASP_SE.jl")

### Set the parameters of GLE ###
Prior = :BPSK_N

N = 1000

# a = 6.0
# a = 5.0
# a = 4.0
# a = 3.0
a = 2.0
# a = 1.0

Seed = 0
# Seed = 100

Ax_T = abs(1.0)

# Cx_T = 1e0
# Cx_T = 1e-1
# Cx_T = 1e-2
# Cx_T = 1e-3
# Cx_T = 1e-4
Cx_T = 1e-5

vw_T = 0.01
# vw_T = 0.1
# vw_T = 0.2
# vw_T = 0.3
# vw_T = 0.4
# vw_T = 0.5
# vw_T = 0.6
# vw_T = 0.7
# vw_T = 0.8
# vw_T = 0.9
# vw_T = 1.0

# cor_flag = false
# cor = 0.

cor_flag = true
cor = 0.0
# cor = 0.2
# cor = 0.3
# cor = 0.4
# cor = 0.43
# cor = 0.6
# cor = 0.8

### Set the parameters of AMP, VAMP, ASP and VASP ###
Epochs = 30

L = 4

Sd = Seed

InitX = :Spectral

Epsilon = 1e-3

Mes = 0.95

Ax_F = Ax_T

Cx_F = 0

vw_F = 0.01
# vw_F = 0.1
# vw_F = 0.3
# vw_F = 0.5
# vw_F = 0.7
# vw_F = 0.9

eps = 2e-16

# Verb = 1
Verb = 0

Num = 10

Vamp_t_Mse_List = zeros(Epochs, Num)
Vamp_t_Eps_List = zeros(Epochs, Num)
Vamp_t_Rho_List = zeros(Epochs, Num)
Vamp_t_A_D_List = zeros(Epochs, Num)
Vamp_t_A_F_List = zeros(Epochs, Num)
Vamp_t_A_H_List = zeros(Epochs, Num)
Vamp_t_A_Chi_List = zeros(Epochs, Num)

Asp_f_Mse_List = zeros(Epochs, Num)
Asp_f_Eps_List = zeros(Epochs, Num)
Asp_f_Rho_List = zeros(Epochs, Num)
Asp_f_A_D_List = zeros(Epochs, Num)
Asp_f_A_F_List = zeros(Epochs, Num)
Asp_f_A_H_List = zeros(Epochs, Num)
Asp_f_A_Chi_List = zeros(Epochs, Num)

Vasp_f_Mse_List = zeros(Epochs, Num)
Vasp_f_Eps_List = zeros(Epochs, Num)
Vasp_f_Rho_List = zeros(Epochs, Num)
Vasp_f_A_D_List = zeros(Epochs, Num)
Vasp_f_A_F_List = zeros(Epochs, Num)
Vasp_f_A_H_List = zeros(Epochs, Num)
Vasp_f_A_Chi_List = zeros(Epochs, Num)

Vasp_SE_f_Mse_List = zeros(Epochs, 1)
Vasp_SE_f_Eps_List = zeros(Epochs, 1)
Vasp_SE_f_Rho_List = zeros(Epochs, 1)
Vasp_SE_f_A_D_List = zeros(Epochs, 1)
Vasp_SE_f_A_F_List = zeros(Epochs, 1)
Vasp_SE_f_A_H_List = zeros(Epochs, 1)
Vasp_SE_f_A_Chi_List = zeros(Epochs, 1)

Int_cor_flag = Int(cor_flag)

Name = "Main_Iteration-cf$Int_cor_flag-c$cor-S$Seed-N$N-a$a-L$L-aT$Ax_T-cT$Cx_T-vT$vw_T-aF$Ax_F-cF$Cx_F-vF$vw_F"

rm("./$Name-S_Vamp_t_Df_Final.bson", force = true)

rm("./$Name-S_Asp_f_Df_Final.bson", force = true)

rm("./$Name-S_Vasp_f_Df_Final.bson", force = true)

rm("./$Name-S_Vasp_SE_f_Df_Final.bson", force = true)

rm("./$Name-fig_MSE.pdf", force = true)
rm("./$Name-fig_D.pdf", force = true)
rm("./$Name-fig_F.pdf", force = true)
rm("./$Name-fig_H.pdf", force = true)
rm("./$Name-fig_Chi.pdf", force = true)

d1 = DateTime(now())

println("nthreads = ", Threads.nthreads())

Threads.@threads for i = 1 : Num
# for i = 1 : Num
	println("")
	println("i: ", i)

	### Define the problem ###
	Prob = Problem_Setting(
			"GLE"; Prior = Prior,
			N = N, a = a, Seed = Seed,
			Ax_T = Ax_T,
			Cx_T = Cx_T,
			vw_T = vw_T,
			cor_flag = cor_flag,
			cor = cor,
			eps = eps
		)

	# ### Solve the problem by VAMP, ASP and VASP ###
	println("VAMP-t")
	Vamp_t_Df, Vamp_t, Vamp_t_Ok, Vamp_t_Prms = Module_VAMP.Solve(Prob,
			Epochs = Epochs,
			Sd = Sd,
			InitX = InitX,
			Epsilon = Epsilon,
			Mes = Mes,
			Ax_T = Ax_T,
			Cx_T = Cx_T,
			vw_T = vw_T,
			eps = eps,
			Verb = Verb,
			Name = Name
		)

	Vamp_t_Mse_List[:, i] .= Vamp_t_Df.Mse
	Vamp_t_Eps_List[:, i] .= Vamp_t_Df.Eps
	Vamp_t_Rho_List[:, i] .= Vamp_t_Df.Rho
	Vamp_t_A_D_List[:, i] .= Vamp_t_Df.A_D
	Vamp_t_A_F_List[:, i] .= Vamp_t_Df.A_F
	Vamp_t_A_H_List[:, i] .= Vamp_t_Df.A_H
	Vamp_t_A_Chi_List[:, i] .= Vamp_t_Df.A_Chi

	println("ASP-f")
	Asp_f_Df, Asp_f, Asp_f_Ok, Asp_f_Prms = Module_ASP.Solve(Prob,
			Epochs = Epochs,
			L = L,
			Sd = Sd,
			InitX = InitX,
			Epsilon = Epsilon,
			Mes = Mes,
			Ax_F = Ax_F,
			Cx_F = Cx_F,
			vw_F = vw_F,
			eps = eps,
			Verb = Verb,
			Name = Name
		)

	Asp_f_Mse_List[:, i] .= Asp_f_Df.Mse
	Asp_f_Eps_List[:, i] .= Asp_f_Df.Eps
	Asp_f_Rho_List[:, i] .= Asp_f_Df.Rho
	Asp_f_A_D_List[:, i] .= Asp_f_Df.A_D
	Asp_f_A_F_List[:, i] .= Asp_f_Df.A_F
	Asp_f_A_H_List[:, i] .= Asp_f_Df.A_H
	Asp_f_A_Chi_List[:, i] .= Asp_f_Df.A_Chi

	println("VASP-f")
	Vasp_f_Df, Vasp_f, Vasp_f_Ok, Vasp_f_Prms = Module_VASP.Solve(Prob,
			Epochs = Epochs,
			L = L,
			Sd = Sd,
			InitX = InitX,
			Epsilon = Epsilon,
			Mes = Mes,
			Ax_F = Ax_F,
			Cx_F = Cx_F,
			vw_F = vw_F,
			eps = eps,
			Verb = Verb,
			Name = Name
		)

	Vasp_f_Mse_List[:, i] .= Vasp_f_Df.Mse
	Vasp_f_Eps_List[:, i] .= Vasp_f_Df.Eps
	Vasp_f_Rho_List[:, i] .= Vasp_f_Df.Rho
	Vasp_f_A_D_List[:, i] .= Vasp_f_Df.A_D
	Vasp_f_A_F_List[:, i] .= Vasp_f_Df.A_F
	Vasp_f_A_H_List[:, i] .= Vasp_f_Df.A_H
	Vasp_f_A_Chi_List[:, i] .= Vasp_f_Df.A_Chi

	# println("VASP-SE-f")
	if i == Num
		Vasp_SE_f_Df, Vasp_SE_f, Vasp_SE_f_Ok, Vasp_SE_f_Prms = Module_VASP_SE.Solve(Prob,
				Epochs = Epochs,
				L = L,
				Sd = Sd,
				InitX = InitX,
				Epsilon = Epsilon,
				Mes = Mes,
				Ax_F = Ax_F,
				Cx_F = Cx_F,
				vw_F = vw_F,
				eps = eps,
				Verb = Verb,
				Name = Name
			)

		Vasp_SE_f_Mse_List[:, 1] .= Vasp_SE_f_Df.Mse
		Vasp_SE_f_Eps_List[:, 1] .= Vasp_SE_f_Df.Eps
		Vasp_SE_f_Rho_List[:, 1] .= Vasp_SE_f_Df.Rho
		Vasp_SE_f_A_D_List[:, 1] .= Vasp_SE_f_Df.A_D
		Vasp_SE_f_A_F_List[:, 1] .= Vasp_SE_f_Df.A_F
		Vasp_SE_f_A_H_List[:, 1] .= Vasp_SE_f_Df.A_H
		Vasp_SE_f_A_Chi_List[:, 1] .= Vasp_SE_f_Df.A_Chi
	end
end

Vamp_t_Df_Final = Df_Push(
		Epochs, Vamp_t_Mse_List, Vamp_t_Eps_List,
		Vamp_t_Rho_List,
		Vamp_t_A_D_List, Vamp_t_A_F_List, Vamp_t_A_H_List, Vamp_t_A_Chi_List
	)
println("-----  VAMP-t  -----")
println(Vamp_t_Df_Final)
BSON.@save "./$Name-S_Vamp_t_Df_Final.bson" Vamp_t_Df_Final

Asp_f_Df_Final = Df_Push(
		Epochs, Asp_f_Mse_List, Asp_f_Eps_List,
		Asp_f_Rho_List,
		Asp_f_A_D_List, Asp_f_A_F_List, Asp_f_A_H_List, Asp_f_A_Chi_List
	)
println("-----  ASP-f  -----")
println(Asp_f_Df_Final)
BSON.@save "./$Name-S_Asp_f_Df_Final.bson" Asp_f_Df_Final

Vasp_f_Df_Final = Df_Push(
		Epochs, Vasp_f_Mse_List, Vasp_f_Eps_List,
		Vasp_f_Rho_List,
		Vasp_f_A_D_List, Vasp_f_A_F_List, Vasp_f_A_H_List, Vasp_f_A_Chi_List
	)
println("-----  VASP-f  -----")
println(Vasp_f_Df_Final)
BSON.@save "./$Name-S_Vasp_f_Df_Final.bson" Vasp_f_Df_Final

Vasp_SE_f_Df_Final = Df_Push(
		Epochs, Vasp_SE_f_Mse_List, Vasp_SE_f_Eps_List,
		Vasp_SE_f_Rho_List,
		Vasp_SE_f_A_D_List, Vasp_SE_f_A_F_List, Vasp_SE_f_A_H_List, Vasp_SE_f_A_Chi_List
	)
println("-----  VASP-SE-f  -----")
println(Vasp_SE_f_Df_Final)
BSON.@save "./$Name-S_Vasp_SE_f_Df_Final.bson" Vasp_SE_f_Df_Final

d2 = DateTime(now())
println("Times: ", d2 - d1)

Epochs_List = [1 : Epochs ...]

Mse_List = zeros(Epochs, 4)

Mse_List[:, 1] .= 10 * log10.(Vamp_t_Df_Final.Mse)

Mse_List[:, 2] .= 10 * log10.(Asp_f_Df_Final.Mse)

Mse_List[:, 3] .= 10 * log10.(Vasp_f_Df_Final.Mse)

Mse_List[:, 4] .= 10 * log10.(Vasp_SE_f_Df_Final.Mse)


fig_MSE = plot(
		Epochs_List, Mse_List,
		title = "VAMP-ASP-VASP",
		xlabel = "Iter", ylabel = "MSE",
		label = ["VAMP-t" "ASP-f" "VASP-f" "VASP-SE-f"]
	)
savefig(fig_MSE, "./$Name-fig_MSE.pdf")

D_List = zeros(Epochs, 2)
D_List[:, 1] .= Vasp_f_Df_Final.A_D
D_List[:, 2] .= Vasp_SE_f_Df_Final.A_D
fig_D = plot(
		Epochs_List, D_List,
		title = "VASP: Algo vs SE",
		xlabel = "Iter", ylabel = "D",
		label = ["D" "SE-D"]
	)
savefig(fig_D, "./$Name-fig_D.pdf")

F_List = zeros(Epochs, 2)
F_List[:, 1] .= Vasp_f_Df_Final.A_F
F_List[:, 2] .= Vasp_SE_f_Df_Final.A_F
fig_F = plot(
		Epochs_List, F_List,
		title = "VASP: Algo vs SE",
		xlabel = "Iter", ylabel = "F",
		label = ["F" "SE-F"]
	)
savefig(fig_F, "./$Name-fig_F.pdf")

H_List = zeros(Epochs, 2)
H_List[:, 1] .= Vasp_f_Df_Final.A_H
H_List[:, 2] .= Vasp_SE_f_Df_Final.A_H
fig_H = plot(
		Epochs_List, H_List,
		title = "VASP: Algo vs SE",
		xlabel = "Iter", ylabel = "H",
		label = ["H" "SE-H"]
	)
savefig(fig_H, "./$Name-fig_H.pdf")

Chi_List = zeros(Epochs, 2)
Chi_List[:, 1] .= Vasp_f_Df_Final.A_Chi
Chi_List[:, 2] .= Vasp_SE_f_Df_Final.A_Chi
fig_Chi = plot(
		Epochs_List, Chi_List,
		title = "VASP: Algo vs SE",
		xlabel = "Iter", ylabel = "Chi",
		label = ["Chi" "SE-Chi"]
	)
savefig(fig_Chi, "./$Name-fig_Chi.pdf")