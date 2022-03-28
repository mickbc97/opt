#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:51:09 2022

@author: Mick
"""

import csv
from gurobipy import *
import pandas as pd
import time

# Import csvs to dataframes
nodedf = pd.read_csv('C:/Users/mickb/OneDrive - University of Oklahoma/School Documents/Thesis/SwRail_Interdiction_Mick/Input/chosen_network_nodes.csv') # nodes
linkdf = pd.read_csv('C:/Users/mickb/OneDrive - University of Oklahoma/School Documents/Thesis/SwRail_Interdiction_Mick/Input/chosen_network_links.csv') # links
# intddf = pd.read_csv('C:/Users/mickb/OneDrive - University of Oklahoma/School Documents/Thesis/SwRail_Interdiction_Mick/Input/interdiction_list.csv') # selected interdiction links
# met_demdf = pd.read_csv('C:/Users/mickb/OneDrive - University of Oklahoma/School Documents/Thesis/SwRail_Interdiction_Mick/Input/interdicted_met_dems.csv') # met demand values
met_demdf = pd.read_csv('C:/Users/mickb/OneDrive - University of Oklahoma/School Documents/Thesis/SwRail_Interdiction_Mick/Input/interdicted_met_dems5.csv') # met demand values

# nodedf = pd.read_csv('/Users/Mick/OneDrive - University of Oklahoma/School Documents/Thesis/SwRail_Interdiction_Mick/mick_network_nodes_full.csv')
# linkdf = pd.read_csv('/Users/Mick/OneDrive - University of Oklahoma/School Documents/Thesis/SwRail_Interdiction_Mick/mick_network_edges_full.csv')

# Create sets of commodities, nodes, links, and interdicted links
K = list(range(0, 15))
N = []
A = []
L = []
L1 = []
#L = [(9,8),(10,9),(1002,1001),(1026,1002),(1001,1000)]

# Assign supply/demand to each node by commodity
# Node:commodity:supply value
temp_supply = {}
b = {}

for i in range(0,len(nodedf)):
    N.append(nodedf.loc[i][0])
    temp_supply = {}
    for k in range(3,18):
        temp_supply[k-3] = nodedf.loc[i][k]/1000000
    b[N[i]] = temp_supply

# Also make a tuplelist version
tuple_N = tuplelist(N.copy())

# # Add unmet demand values to dictionary with keys: li, lj, k, j and value: met demand
# met_dem = {}
# for i in range(0,len(met_demdf)):
#     met_dem[int(met_demdf.loc[i][0]), int(met_demdf.loc[i][1]), int(met_demdf.loc[i][2]), int(met_demdf.loc[i][3])] = met_demdf.loc[i][4]
# met_demdf[(met_demdf['li']==li) & (met_demdf['lj']==lj) & (met_demdf['k']==k) & (met_demdf['j']==j)]['met_demand'].values[0]
# li = 1085
# lj = 311
# k = 0
# j =20
# Add links to set from csv file and add capacity to dictionary
# (node i, node j)
# Link:commodity:capacity value
temp_capacity = {}
c = {}

for i in range(0,len(linkdf)):
    A.append(tuple((linkdf.loc[i][0],linkdf.loc[i][1])))
    temp_capacity = {}
    for k in range(2,17):
        temp_capacity[k-2] = linkdf.loc[i][k]/1000000
    c[A[i]] = temp_capacity

# Also create tuplelist for arcs for constraints
tuple_A = tuplelist(A.copy())

# # Create list of all interdicted links
# for i in range(0,len(intddf)):
#     L1.append(tuple((int(intddf.loc[i][0]),int(intddf.loc[i][1]))))

# # Choose the top 100 of interdicted links
# for i in range(0,100):
#     L.append(L1[i])

# Duplicate links to interdicted links set
interdDL = A.copy()
interdicted = []

# Loop through each edge and find match ij = ji, create nested list
while len(interdDL) > 0:
    edgeTL = interdDL.pop()
    
    for i in range(len(interdDL)):
        edgeTLi = interdDL[i]
        
        if edgeTL[0] == edgeTLi[1] and edgeTL[1] == edgeTLi[0]:
            comb = [edgeTL, edgeTLi]
            interdicted.append(comb)
            interdDL.pop(i)
            
            break

# Add undirected links to interdicted links set ********for i in range(0,len(interdicted))
for i in range(0,len(interdicted)):
    L.append(interdicted[i][0])

# Create sets of demand, supply, and transshipment nodes for each commodity
temp_demand_nodes = {}
D = {}
temp_supply_nodes = {}
S = {}
temp_trans_nodes = {}
T = {}

for k in range(0,len(K)):
    temp_demand_nodes = {}
    temp_supply_nodes = {}
    temp_trans_nodes = {}
    for i in range(1, len(b)+1):
        if b[i][k] < 0:
            temp_demand_nodes[i] = -b[i][k]
        elif b[i][k] > 0:
            temp_supply_nodes[i] = b[i][k]
        elif b[i][k] == 0:
            temp_trans_nodes[i] = b[i][k]
    D[k] = temp_demand_nodes
    S[k] = temp_supply_nodes
    T[k] = temp_trans_nodes

# Create dict of just the demand nodes for each commodity
D_keys = {}
S_keys = {}

for k in K:
    D_keys[k] = tuplelist(list(D[k].keys()))
    S_keys[k] = tuplelist(list(D[k].keys()))

# Create dict of total demand for commodities
Dk = {}
for k in K:
    summed = 0
    for j in D[k].keys():
        summed += D[k][j]
    Dk[k] = summed

# Create lists of tuples of demand/supply/transshipment nodes and commodities
D_with_k = []
S_with_k = []
T_with_k = []

for k in K:
    for j in D[k]:
        D_with_k.append(tuple((j, k)))
    for j in S[k]:
        S_with_k.append(tuple((j, k)))
    for j in T[k]:
        T_with_k.append(tuple((j, k)))

# Make them tuplelists
D_with_k = tuplelist(D_with_k)
S_with_k = tuplelist(S_with_k)
T_with_k = tuplelist(T_with_k)

Ak = []
# Create links with commodities for unmet demand LinExpr
for k in K:
    for i in range(0, len(A)):
        Ak.append(tuple((A[i][0], A[i][1], k)))

Ak = tuplelist(Ak)

# Create list of maximum allowable percentage increase in capacity for commodities
p = [0.00430, 0.00164, 0.00063, 0.00271, 0.00650, 0.00121, 0.00055, 0.00030, 0.00037,\
     0.00648, 0.00032, 0.00071, 0.00421, 0.00059, 0.00023]

# Create list of max added capacity - set to 5% of total commodity supply for now
alpha_max = []
for k in K:
    keys = list(S[k].keys())
    sum_supply = 0
    for i in keys:
        sum_supply += S[k][i]
    alpha_max.append(round(sum_supply*(p[k]/2), 6))

# -------------------------------------------------------------------------------------------------
# Set initial time
t1 = time.process_time()

# Create optimization model
m = Model('bal_net_flow')
t00 = time.process_time()
# Decision variables - flow and unmet demand for each commodity
x = m.addVars(A, K, L, name = "x") #flow variables
alpha = m.addVars(A, K, name = "alpha") # added capacity variables
# n_s = m.addVars(N, K, L, name = "n_s") # maximum supply value variables
# n_d = m.addVars(N, K, L, name = "n_d") # maximum demand value variables
U = m.addVars(L, K, name = "U") # imaginary variable for objective linearization
lin = m.addVars(K, name = "lin")
I = m.addVars(L, K, name = "I")

m.update()
t01 = time.process_time()
print('Variables declared', t01-t00)
t00 = time.process_time()
# Equation 12: Capacity constraint for link (i,j) for commodity k
m.addConstrs(
    (x[i, j, k, li, lj] <= (1 + alpha[i, j, k]) * c[i, j][k] for i, j in A for k in K for li, lj in L), "cap")

# m.addConstrs(x[i,j,k, li, lj] >= 0 for i,j in A for k in K for li, lj in L)

# Equation 13: Added capacity constraint
m.addConstrs(
    (alpha[i, j, k] <= p[k] for i, j in A for k in K), "p^k"
    )
t01 = time.process_time()
print('Capacity constraints done', t01-t00)
m.addConstrs(
    (quicksum(alpha[i, j, k] * c[i, j][k] for i, j in tuple_A.select('*', '*')) <= alpha_max[k] for k in K), "max_alpha"
    )

# Equations 15-17: Flow balance constraints for link (i,j) for commodity k
t00 = time.process_time()
# m.addConstrs(
#     (quicksum(x[i1, j, k, li, lj] for i1, j in tuple_A.select('*', j)) -
#     quicksum(x[j, i2, k, li, lj] for j, i2 in tuple_A.select(j, '*')) ==
#     n_d[j, k, li, lj] for j, k in D_with_k for li, lj in L)
#     )

m.addConstrs(
    (quicksum(x[i1, j, k, li, lj] for i1, j in tuple_A.select('*', j)) -
    quicksum(x[j, i2, k, li, lj] for j, i2 in tuple_A.select(j, '*')) <=
    D[k][j] for j, k in D_with_k for li, lj in L)
    )
m.addConstrs(
    (quicksum(x[i1, j, k, li, lj] for i1, j in tuple_A.select('*', j)) -
    quicksum(x[j, i2, k, li, lj] for j, i2 in tuple_A.select(j, '*')) >=
    0 for j, k in D_with_k for li, lj in L)
    )
# m.addConstrs(
#     (quicksum(x[i1, j, k, li, lj] for i1, j in tuple_A.select('*', j)) -
#     quicksum(x[j, i2, k, li, lj] for j, i2 in tuple_A.select(j, '*')) >=
#     met_demdf[(met_demdf['li']==li) & (met_demdf['lj']==lj) & (met_demdf['k']==k) & (met_demdf['j']==j)]['met_demand'].values[0] for j, k in D_with_k for li, lj in L)
#     )

# m.addConstrs(
#     (-quicksum(x[i1, j, k, li, lj] for i1, j in tuple_A.select('*', j)) +
#     quicksum(x[j, i2, k, li, lj] for j, i2 in tuple_A.select(j, '*')) ==
#     n_s[j, k, li, lj] for j, k in S_with_k for li, lj in L)
#     )

m.addConstrs(
    (-quicksum(x[i1, j, k, li, lj] for i1, j in tuple_A.select('*', j)) +
    quicksum(x[j, i2, k, li, lj] for j, i2 in tuple_A.select(j, '*')) <=
    S[k][j] for j, k in S_with_k for li, lj in L)
    )
m.addConstrs(
    (-quicksum(x[i1, j, k, li, lj] for i1, j in tuple_A.select('*', j)) +
    quicksum(x[j, i2, k, li, lj] for j, i2 in tuple_A.select(j, '*')) >=
    0 for j, k in S_with_k for li, lj in L)
    )

m.addConstrs(
    (quicksum(x[i1, j, k, li, lj] for i1, j in tuple_A.select('*', j)) -
    quicksum(x[j, i2, k, li, lj] for j, i2 in tuple_A.select(j, '*')) ==
    0 for j, k in T_with_k for li, lj in L)
    )
t01 = time.process_time()
print('First flow balance constraints done', t01-t00)
t00 = time.process_time()

# Added constraints for modeling - let variables max be supply/demand at node j for commodity k
# m.addConstrs(
#     (n_s[j, k, li, lj] <= S[k][j] for j, k in S_with_k for li, lj in L)
#     )
# m.addConstrs(
#     (n_s[j, k, li, lj] >= 0 for j, k in S_with_k for li, lj in L)
#     )
# m.addConstrs(
#     (n_d[j, k, li, lj] <= D[k][j] for j, k in D_with_k for li, lj in L)
#     )
# m.addConstrs(
#     (n_d[j, k, li, lj] >= met_demdf[(met_demdf['li']==li) & (met_demdf['lj']==lj) & (met_demdf['k']==k) & (met_demdf['j']==j)]['met_demand'].values[0] for j, k in D_with_k for li, lj in L)
#     )
# m.addConstrs(
#     (n_d[j, k, li, lj] >= 0 for j, k in D_with_k for li, lj in L)
#     )
t01 = time.process_time()
print('Second flow balance constraints done', t01-t00)
t00 = time.process_time()
# Equations 18-19: Interdiction constraints
m.addConstrs(
    (x[li, lj, k, li, lj] == 0 for k in K for li, lj in L), "interdicted_1")

m.addConstrs(
    (x[lj, li, k, li, lj] == 0 for k in K for li, lj in L), "interdicted_2")
t01 = time.process_time()
print('Interdiction constraints done', t01-t00)
t00 = time.process_time()
# Z1 = quicksum(((Dk[k] - quicksum(n_d[j, k, li, lj] for j in D[k].keys())) /
#                     Dk[k]) for k in K for li, lj in L)

# quicksum((D[k][j] - (quicksum(x[i, j, k, li, lj] for i, j in A.select('*', j)) - quicksum()))/D[k][j] for j in D[k].keys() for k in K for li, lj in L)
m.addConstrs(
    (quicksum((D[k][j] - (quicksum(x[i1, j, k, li, lj] for i1, j in tuple_A.select('*', j)) -
    quicksum(x[j, i2, k, li, lj] for j, i2 in tuple_A.select(j, '*'))))/D[k][j] for j,k in D_with_k) <= unmet_demand[li,lj] for li, lj in L)
    )

m.addConstrs(
    (lin[k] == 1/(len(L)) * quicksum((Dk[k] - (quicksum(x[i1, j, k, li, lj] for i1, j in tuple_A.select('*', j)) -
    quicksum(x[j, i2, k, li, lj] for j, i2 in tuple_A.select(j, '*')))) / Dk[k] for j in D[k].keys() for li, lj in L) for k in K)
    )

# Equations 8-9: To transform from absolute value to linear constraints
m.addConstrs(
    (quicksum((Dk[k] - (quicksum(x[i1, j, k, li, lj] for i1, j in tuple_A.select('*', j)) -
    quicksum(x[j, i2, k, li, lj] for j, i2 in tuple_A.select(j, '*')))) / Dk[k] for j1 in D[k].keys()) - lin[k] <= U[li, lj, k] for k in K for li, lj in L)
    )

# m.addConstrs(
#     (quicksum((Dk[k] - (quicksum(x[i1, j, k, li1, lj1] for i1, j in tuple_A.select('*', j)) -
#     quicksum(x[j, i2, k, li1, lj1] for j, i2 in tuple_A.select(j, '*')))) / Dk[k] for j1 in D[k].keys()) - 1/(len(L)) * quicksum((Dk[k] - (quicksum(x[i1, j, k, li2, lj2] for i1, j in tuple_A.select('*', j)) -
#     quicksum(x[j, i2, k, li2, lj2] for j, i2 in tuple_A.select(j, '*')))) / Dk[k] for j1 in D[k].keys() for li2, lj2 in L) <= U[li1, lj1, k] for k in K for li1, lj1 in L)
#     )
t01 = time.process_time()
print('First linearization constraint done', t01-t00)
t00 = time.process_time()
m.addConstrs(
    (-(quicksum((Dk[k] - (quicksum(x[i1, j, k, li, lj] for i1, j in tuple_A.select('*', j)) -
    quicksum(x[j, i2, k, li, lj] for j, i2 in tuple_A.select(j, '*')))) / Dk[k] for j1 in D[k].keys()) - lin[k]) <= U[li, lj, k] for k in K for li, lj in L)
    )

# m.addConstrs(
#     (-(quicksum((Dk[k] - (quicksum(x[i1, j, k, li1, lj1] for i1, j in tuple_A.select('*', j)) -
#     quicksum(x[j, i2, k, li1, lj1] for j, i2 in tuple_A.select(j, '*')))) / Dk[k] for j1 in D[k].keys()) - 1/(len(L)) * quicksum((Dk[k] - (quicksum(x[i1, j, k, li2, lj2] for i1, j in tuple_A.select('*', j)) -
#     quicksum(x[j, i2, k, li2, lj2] for j, i2 in tuple_A.select(j, '*')))) / Dk[k] for j1 in D[k].keys() for li2, lj2 in L)) <= U[li1, lj1, k] for k in K for li1, lj1 in L)
#     )
t01 = time.process_time()
print('Second linearization constraint done', t01-t00)

m.update()

# Equation 7: Objective function
Z = quicksum(U[li, lj, k] for li, lj in L for k in K)

# Set objective
m.setObjective(Z, GRB.MINIMIZE)

m.update()

# Solve the model    
m.optimize()

# Note time taken
t2 = time.process_time()

# Note time taken, objective value
result_list = []
result_list.append('Time taken')
result_list.append(t2-t1)
result_list.append('Objective value')
result_list.append(m.objVal)

# Add met demand values to dict
result_dict = {}
for k in K:
    for li, lj in L:
        summed = 0
        for j in D[k].keys():
            summed += n_d[j,k, li, lj].X
        sumd = summed
        result_dict[k] = sumd/Dk[k]

# Write file of added demand values
name = 'added_demand_eps.csv'
f = open(name,'w', newline='')
header = ['k', 'i', 'j', 'added_demand']
writer = csv.writer(f)
writer.writerow(header)
for k in K:
    for i, j in A:
        line = [k,i,j, alpha[i, j, k].X]
        writer.writerow(line)
f.close()

