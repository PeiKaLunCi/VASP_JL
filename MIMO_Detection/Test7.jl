include("./Common.jl")

println("")
println("")
println("")

Name = "Main_QQplot-cf1-c0.25-N5000-a1.0-L10-aT1.0-vT0.1-aF1.0-vF0.15"

Gvasp_Df_List = BSON.load("$Name-Gvasp_Df_List.bson")
Gvasp_Df_List = Gvasp_Df_List[:Gvasp_Df_List]
println("Gvasp_Df_List: ", Gvasp_Df_List)

S_m_s_z_Df_List = BSON.load("$Name-Gvasp_S_m_s_z_Df_List")
S_m_s_z_Df_List = S_m_s_z_Df_List[:Gvasp_S_m_s_z_Df_List]
println("S_m_s_z_Df_List: ", size(S_m_s_z_Df_List))

S_m_s_x_Df_List = BSON.load("$Name-Gvasp_S_m_s_x_Df_List")
S_m_s_x_Df_List = S_m_s_x_Df_List[:Gvasp_S_m_s_x_Df_List]
println("S_m_s_x_Df_List: ", size(S_m_s_x_Df_List))

S_m_p_x_Df_List = BSON.load("$Name-Gvasp_S_m_p_x_Df_List")
S_m_p_x_Df_List = S_m_p_x_Df_List[:Gvasp_S_m_p_x_Df_List]
println("S_m_p_x_Df_List: ", size(S_m_p_x_Df_List))

S_m_p_z_Df_List = BSON.load("$Name-Gvasp_S_m_p_z_Df_List")
S_m_p_z_Df_List = S_m_p_z_Df_List[:Gvasp_S_m_p_z_Df_List]
println("S_m_p_z_Df_List: ", size(S_m_p_z_Df_List))

x0_List = BSON.load("$Name-x0_List")
x0_List = x0_List[:x0_List]
println("x0_List: ", size(x0_List))

z_List = BSON.load("$Name-z_List")
z_List = z_List[:z_List]
println("z_List: ", size(z_List))

M, Epochs, Num = size(S_m_s_z_Df_List)
N, Epochs, Num = size(S_m_s_x_Df_List)
println("M: ", M, ", N: ", N, ", Epochs: ", Epochs, ", Num: ", Num)

println("---------------------------------")
rm("./$Name-Test7.txt", force = true)
io = open("./$Name-Test7.txt", "w")

H = sqrt(1 / N) * randn(M, N)

H_svd = svd(H, full = true)
U = H_svd.U
Vt = H_svd.Vt
V = Vt'

Idx = Epochs

SX_1 = S_m_s_x_Df_List[:, Idx, :] / Gvasp_Df_List.S_v_s_x[Idx] .+ Gvasp_Df_List.S_HD_1x[Idx] * x0_List
vec = abs.(SX_1)
println("SX_1: ", [maximum(vec), minimum(vec), mean(vec)])

SX_2 = SX_1[:]' / sqrt(- Gvasp_Df_List.S_HF_1x[Idx])
println(io, "SX_2: ", SX_2)
vec = abs.(SX_2)
println("SX_2: ", [maximum(vec), minimum(vec), mean(vec)])

SZ_1 = S_m_s_z_Df_List[:, Idx, :] / Gvasp_Df_List.S_v_s_z[Idx] .+ Gvasp_Df_List.S_HD_2z[Idx] * z_List
vec = abs.(SZ_1)
println("SZ_1: ", [maximum(vec), minimum(vec), mean(vec)])

SZ_2 = SZ_1[:]' / sqrt(- Gvasp_Df_List.S_HF_2z[Idx])
println(io, "SZ_2: ", SZ_2)
vec = abs.(SZ_2)
println("SZ_2: ", [maximum(vec), minimum(vec), mean(vec)])

println(norm(S_m_p_x_Df_List[:, Idx, :] .- x0_List))
println(io, "S_m_p_x_Df_List: ", S_m_p_x_Df_List[:, Idx, :])
println(io, "x0_List: ", x0_List)

PX_0 = S_m_p_x_Df_List[:, Idx, :] / Gvasp_Df_List.S_v_p_x[Idx] .+ Gvasp_Df_List.S_HD_2x[Idx] * x0_List
vec = abs.(PX_0)
println("PX_0: ", [maximum(vec), minimum(vec), mean(vec)])

PX_1 = zeros(N, Num)
for i = 1 : Num
	PX_1[:, i] .= V' * PX_0[:, i]
end
vec = abs.(PX_1)
println("PX_1: ", [maximum(vec), minimum(vec), mean(vec)])

PX_2 = PX_1[:]' / sqrt(- Gvasp_Df_List.S_HF_2x[Idx])
println(io, "PX_2: ", PX_2)
vec = abs.(PX_2)
println("PX_2: ", [maximum(vec), minimum(vec), mean(vec)])

PX_3 = zeros(N, Num)
PX_4 = zeros(N, Num)

PX_3 = S_m_p_x_Df_List[:, Idx, :] / Gvasp_Df_List.S_v_p_x[Idx]
PX_3 = PX_3[:]'
println(io, "PX_3: ", PX_3)

for i = 1 : Num
	PX_4[:, i] .= - Gvasp_Df_List.S_HD_2x[Idx] * x0_List[:, i] + sqrt(- Gvasp_Df_List.S_HF_2x[Idx]) * V * randn(N)
end

PX_4 = PX_4[:]'
println(io, "PX_4: ", PX_4)

PX_5 = zeros(N, Num)
PX_6 = zeros(N, Num)

for i = 1 : Num
	PX_5[:, i] .= V' * S_m_p_x_Df_List[:, Idx, i] / Gvasp_Df_List.S_v_p_x[Idx]
	PX_6[:, i] .= - Gvasp_Df_List.S_HD_2x[Idx] * V' * x0_List[:, i] + sqrt(- Gvasp_Df_List.S_HF_2x[Idx]) * randn(N)
end

PX_5 = PX_5[:]'
println(io, "PX_5: ", PX_5)

PX_6 = PX_6[:]'
println(io, "PX_6: ", PX_6)

println(norm(S_m_p_z_Df_List[:, Idx, :] .- z_List))
println(io, "S_m_p_z_Df_List: ", S_m_p_z_Df_List[:, Idx, :])
println(io, "z_List: ", z_List)

PZ_0 = S_m_p_z_Df_List[:, Idx, :] / Gvasp_Df_List.S_v_p_z[Idx] .+ Gvasp_Df_List.S_HD_1z[Idx] * z_List
vec = abs.(PZ_0)
println("PZ_0: ", [maximum(vec), minimum(vec), mean(vec)])

PZ_1 = zeros(M, Num)
for i = 1 : Num
	PZ_1[:, i] .= U' * PZ_0[:, i]
end
vec = abs.(PZ_1)
println("PZ_1: ", [maximum(vec), minimum(vec), mean(vec)])

PZ_2 = PZ_1[:]' / sqrt(- Gvasp_Df_List.S_HF_1z[Idx]) 
vec = abs.(PZ_2)
println(io, "PZ_2: ", PZ_2)
println("PZ_2: ", [maximum(vec), minimum(vec), mean(vec)])

PZ_3 = zeros(M, Num)
PZ_4 = zeros(M, Num)

PZ_3 = S_m_p_z_Df_List[:, Idx, :] / Gvasp_Df_List.S_v_p_z[Idx]
PZ_3 = PZ_3[:]'
println(io, "PZ_3: ", PZ_3)

for i = 1 : Num
	PZ_4[:, i] .= - Gvasp_Df_List.S_HD_1z[Idx] * z_List[:, i] + sqrt(- Gvasp_Df_List.S_HF_1z[Idx]) * U * randn(M)
end

PZ_4 = PZ_4[:]'
println(io, "PZ_4: ", PZ_4)

PZ_5 = zeros(M, Num)
PZ_6 = zeros(M, Num)

for i = 1 : Num
	PZ_5[:, i] .= U' * S_m_p_z_Df_List[:, Idx, i] / Gvasp_Df_List.S_v_p_z[Idx]
	PZ_6[:, i] .= - Gvasp_Df_List.S_HD_1z[Idx] * U' * z_List[:, i] + sqrt(- Gvasp_Df_List.S_HF_1z[Idx]) * randn(M)
end

PZ_5 = PZ_5[:]'
println(io, "PZ_5: ", PZ_5)

PZ_6 = PZ_6[:]'
println(io, "PZ_6: ", PZ_6)

close(io)