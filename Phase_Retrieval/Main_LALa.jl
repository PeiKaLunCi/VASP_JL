# run code in a terminal of Ubuntu -> export JULIA_NUM_THREADS=4
include("./Common.jl")
include("./GASP.jl")
include("./GVASP.jl")
include("./GVASP_SE.jl")

### Set the parameters of GLE ###
Prior = :Gauss

N = 1000

a = 8

Seed = 0

C_x = 1.
Mask = 1.
vw = 0.
Act = abs

# cor_flag = false
# cor = 0.

cor_flag = true
cor = 0.4

### Set the parameters of GASP and GVASP ###
Epochs = 30

L_List = [10, 20, 100]

La_List = [1e-5, 1e-4, 1e-3, 1e-2]
Drop_La = false

Sd = 0

InitX = :Spectral

Ro = 1e-3

Epsilon = 1e-3

Mes = 1.0

va = 1 / 2
# va = 1.

EPS = 2e-16

# Verb = 1
Verb = 0

Num = 20

size_L = size(L_List, 1)
size_La = size(La_List, 1)

Gasp_Mse_LALa_List = zeros(size_L, size_La)
Gasp_Eps_LALa_List = zeros(size_L, size_La)
Gasp_Rho_LALa_List = zeros(size_L, size_La)
Gasp_A_D_LALa_List = zeros(size_L, size_La)
Gasp_A_F_LALa_List = zeros(size_L, size_La)
Gasp_A_H_LALa_List = zeros(size_L, size_La)
Gasp_A_GsF_LALa_List = zeros(size_L, size_La)

Gvasp_Mse_LALa_List = zeros(size_L, size_La)
Gvasp_Eps_LALa_List = zeros(size_L, size_La)
Gvasp_Rho_LALa_List = zeros(size_L, size_La)
Gvasp_A_D_LALa_List = zeros(size_L, size_La)
Gvasp_A_F_LALa_List = zeros(size_L, size_La)
Gvasp_A_H_LALa_List = zeros(size_L, size_La)
Gvasp_A_GsF_LALa_List = zeros(size_L, size_La)

Gvasp_SE_Mse_LALa_List = zeros(size_L, size_La)
Gvasp_SE_Eps_LALa_List = zeros(size_L, size_La)
Gvasp_SE_Rho_LALa_List = zeros(size_L, size_La)
Gvasp_SE_A_D_LALa_List = zeros(size_L, size_La)
Gvasp_SE_A_F_LALa_List = zeros(size_L, size_La)
Gvasp_SE_A_H_LALa_List = zeros(size_L, size_La)
Gvasp_SE_A_GsF_LALa_List = zeros(size_L, size_La)

Int_cor_flag = Int(cor_flag)
Name = "Main_LALa-cf$Int_cor_flag-c$cor-N$N-a$a-LL$L_List-LaL$La_List-va$va"

rm("./$Name-Gasp_LALa_Final", force = true)
rm("./$Name-Gvasp_LALa_Final", force = true)
rm("./$Name-Gvasp_SE_LALa_Final", force = true)

rm("./$Name-fig_MSE.pdf", force = true)
rm("./$Name-fig_D.pdf", force = true)
rm("./$Name-fig_F.pdf", force = true)
rm("./$Name-fig_H.pdf", force = true)
rm("./$Name-fig_GsF.pdf", force = true)

d1 = DateTime(now())

println("nthreads = ", Threads.nthreads())

for Index_L  = 1 : size_L
	for Index_La = 1 : size_La
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

		Threads.@threads for i = 1 : Num
			println("L[$Index_L]: ", L_List[Index_L])
			println("La[$Index_La]: ", La_List[Index_La])
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
					L = L_List[Index_L, 1],
					La = La_List[Index_La, 1],
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
					L = L_List[Index_L, 1],
					La = La_List[Index_La, 1],
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
					L = L_List[Index_L, 1],
					La = La_List[Index_La, 1],
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

		Gasp_Mse_LALa_List[Index_L, Index_La] = Gasp_Df_Final.Mse[Epochs]
		Gasp_Eps_LALa_List[Index_L, Index_La] = Gasp_Df_Final.Eps[Epochs]
		Gasp_Rho_LALa_List[Index_L, Index_La] = Gasp_Df_Final.Rho[Epochs]
		Gasp_A_D_LALa_List[Index_L, Index_La] = Gasp_Df_Final.A_D[Epochs]
		Gasp_A_F_LALa_List[Index_L, Index_La] = Gasp_Df_Final.A_F[Epochs]
		Gasp_A_H_LALa_List[Index_L, Index_La] = Gasp_Df_Final.A_H[Epochs]
		Gasp_A_GsF_LALa_List[Index_L, Index_La] = Gasp_Df_Final.A_GsF[Epochs]

		Gvasp_Df_Final = Df_Push(
				Epochs, Gvasp_Mse_List, Gvasp_Eps_List,
				Gvasp_Rho_List,
				Gvasp_A_D_List, Gvasp_A_F_List, Gvasp_A_H_List, Gvasp_A_GsF_List
			)

		Gvasp_Mse_LALa_List[Index_L, Index_La] = Gvasp_Df_Final.Mse[Epochs]
		Gvasp_Eps_LALa_List[Index_L, Index_La] = Gvasp_Df_Final.Eps[Epochs]
		Gvasp_Rho_LALa_List[Index_L, Index_La] = Gvasp_Df_Final.Rho[Epochs]
		Gvasp_A_D_LALa_List[Index_L, Index_La] = Gvasp_Df_Final.A_D[Epochs]
		Gvasp_A_F_LALa_List[Index_L, Index_La] = Gvasp_Df_Final.A_F[Epochs]
		Gvasp_A_H_LALa_List[Index_L, Index_La] = Gvasp_Df_Final.A_H[Epochs]
		Gvasp_A_GsF_LALa_List[Index_L, Index_La] = Gvasp_Df_Final.A_GsF[Epochs]

		Gvasp_SE_Df_Final = Df_Push(
				Epochs, Gvasp_SE_Mse_List, Gvasp_SE_Eps_List,
				Gvasp_SE_Rho_List,
				Gvasp_SE_A_D_List, Gvasp_SE_A_F_List, Gvasp_SE_A_H_List, Gvasp_SE_A_GsF_List
			)

		Gvasp_SE_Mse_LALa_List[Index_L, Index_La] = Gvasp_SE_Df_Final.Mse[Epochs]
		Gvasp_SE_Eps_LALa_List[Index_L, Index_La] = Gvasp_SE_Df_Final.Eps[Epochs]
		Gvasp_SE_Rho_LALa_List[Index_L, Index_La] = Gvasp_SE_Df_Final.Rho[Epochs]
		Gvasp_SE_A_D_LALa_List[Index_L, Index_La] = Gvasp_SE_Df_Final.A_D[Epochs]
		Gvasp_SE_A_F_LALa_List[Index_L, Index_La] = Gvasp_SE_Df_Final.A_F[Epochs]
		Gvasp_SE_A_H_LALa_List[Index_L, Index_La] = Gvasp_SE_Df_Final.A_H[Epochs]
		Gvasp_SE_A_GsF_LALa_List[Index_L, Index_La] = Gvasp_SE_Df_Final.A_GsF[Epochs]
	end
end

Gasp_LALa_Final = Dict([
		(:Mse, Gasp_Mse_LALa_List),
		(:Eps, Gasp_Eps_LALa_List),
		(:Rho, Gasp_Rho_LALa_List),
		(:A_D, Gasp_A_D_LALa_List),
		(:A_F, Gasp_A_F_LALa_List),
		(:A_H, Gasp_A_H_LALa_List),
		(:A_GsF, Gasp_A_GsF_LALa_List),
	])
println(Gasp_LALa_Final)
BSON.@save "./$Name-Gasp_LALa_Final" Gasp_LALa_Final

Gvasp_LALa_Final = Dict([
		(:Mse, Gvasp_Mse_LALa_List),
		(:Eps, Gvasp_Eps_LALa_List),
		(:Rho, Gvasp_Rho_LALa_List),
		(:A_D, Gvasp_A_D_LALa_List),
		(:A_F, Gvasp_A_F_LALa_List),
		(:A_H, Gvasp_A_H_LALa_List),
		(:A_GsF, Gvasp_A_GsF_LALa_List),
	])
println(Gvasp_LALa_Final)
BSON.@save "./$Name-Gvasp_LALa_Final" Gvasp_LALa_Final

Gvasp_SE_LALa_Final = Dict([
		(:Mse, Gvasp_SE_Mse_LALa_List),
		(:Eps, Gvasp_SE_Eps_LALa_List),
		(:Rho, Gvasp_SE_Rho_LALa_List),
		(:A_D, Gvasp_SE_A_D_LALa_List),
		(:A_F, Gvasp_SE_A_F_LALa_List),
		(:A_H, Gvasp_SE_A_H_LALa_List),
		(:A_GsF, Gvasp_SE_A_GsF_LALa_List),
	])
println(Gvasp_SE_LALa_Final)
BSON.@save "./$Name-Gvasp_SE_LALa_Final" Gvasp_SE_LALa_Final

d2 = DateTime(now())
println("Times: ", d2 - d1)

La_List = - 10 * log10.(La_List)

Mse_List = [Gasp_Mse_LALa_List; Gvasp_Mse_LALa_List; Gvasp_SE_Mse_LALa_List]
Mse_List = 10 * log10.(Mse_List)
fig_MSE = plot(
		La_List, Mse_List',
		title = "GASP-GVASP",
		xlabel = "La", ylabel = "MSE"
	)
savefig(fig_MSE, "./$Name-fig_MSE.pdf")

D_List = [Gasp_A_D_LALa_List; Gvasp_A_D_LALa_List; Gvasp_SE_A_D_LALa_List]

fig_D = plot(
		La_List, D_List',
		title = "GASP-GVASP",
		xlabel = "La", ylabel = "D"
	)
savefig(fig_D, "./$Name-fig_D.pdf")

F_List = [Gasp_A_F_LALa_List; Gvasp_A_F_LALa_List; Gvasp_SE_A_F_LALa_List]

fig_F = plot(
		La_List, F_List',
		title = "GASP-GVASP",
		xlabel = "La", ylabel = "F"
	)
savefig(fig_F, "./$Name-fig_F.pdf")

H_List = [Gasp_A_H_LALa_List; Gvasp_A_H_LALa_List; Gvasp_SE_A_H_LALa_List]

fig_H = plot(
		La_List, H_List',
		title = "GASP-GVASP",
		xlabel = "La", ylabel = "H"
	)
savefig(fig_H, "./$Name-fig_H.pdf")

GsF_List = [Gasp_A_GsF_LALa_List; Gvasp_A_GsF_LALa_List; Gvasp_SE_A_GsF_LALa_List]

fig_GsF = plot(
		La_List, GsF_List',
		title = "GASP-GVASP",
		xlabel = "La", ylabel = "GsF"
	)
savefig(fig_GsF, "./$Name-fig_GsF.pdf")