#Copyright 2022 Gökalp Çevik

#MIT License
#
#Copyright (c) 2022 Gökalp Çevik
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

# 1-D Infinite potential well problem
# with the well bounds ranging from x=-L/2, x=+L/2, where L is the width
# of the well.

from sympy import init_printing
from sympy import *
import sympy

init_printing()

# Use unicode or not when printing the results.
use_unicode = False

# Particle rest mass
m = sympy.Symbol("m",positive=True)
# Well width
L = sympy.Symbol("L",positive=True)
# Reduced Planck Constant
hbar = sympy.Symbol("hbar",positive=True)
# Position
x = sympy.Symbol("x")
# Time
t = sympy.Symbol("t",positive=True)

# Returns the nth eigenket of the Hamiltonian.
def HamiltonianEigenket(n):
    if n % 2 == 0:
        return sympy.sqrt(2/L)*sympy.sin(n*sympy.pi*x/L)
    else:
        return sympy.sqrt(2/L)*sympy.cos(n*sympy.pi*x/L)

# Returns the nth eigenvalue of the Hamiltonian.
def Energy(n):
    return n**2*sympy.pi**2*hbar**2/(2*m*L**2)

# Time evolution operator.
def TimeEvolution(n):
    return sympy.exp(-sympy.I*Energy(n)*t/hbar)

psi_1 = HamiltonianEigenket(1)
psi_2 = HamiltonianEigenket(2)

# Define the states here --should be normalized--
# in this case it is the superposition of 
# the first eigenket and the second eigenket 
# of the Hamiltonian operator.
state = 1/sympy.sqrt(2)*(psi_1 + psi_2)
state_t = 1/sympy.sqrt(2)*(psi_1*TimeEvolution(1) + psi_2*TimeEvolution(2))

expectation_value_x = sympy.integrate(sympy.conjugate(state)*x*state,(x,-L/2,L/2))
expectation_value_x_t = sympy.integrate(sympy.conjugate(state_t)*x*state_t,(x,-L/2,L/2))

expectation_value_x_squared = sympy.integrate(sympy.conjugate(state)*x**2*state,(x,-L/2,L/2))
expectation_value_x_squared_t = sympy.integrate(sympy.conjugate(state_t)*x**2*state_t,(x,-L/2,L/2))


expectation_value_p = -sympy.I*hbar*sympy.integrate(sympy.conjugate(state)*sympy.diff(state,x),(x,-L/2,L/2))
expectation_value_p_t = -sympy.I*hbar*sympy.integrate(sympy.conjugate(state_t)*sympy.diff(state_t,x),(x,-L/2,L/2))

expectation_value_Energy = -hbar**2/2/m*sympy.integrate(sympy.conjugate(state)*sympy.diff(state,x,x),(x,-L/2,L/2))
expectation_value_Energy_t = -hbar**2/2/m*sympy.integrate(sympy.conjugate(state_t)*sympy.diff(state_t,x,x),(x,-L/2,L/2))

print('Expectation value of x(t=0):')
pprint(expectation_value_x.simplify(),use_unicode=use_unicode)

print('Expectation value of x(t):')
pprint(expectation_value_x_t.simplify(),use_unicode=use_unicode)

print('Expectation value of x^2(t=0):')
pprint(expectation_value_x_squared.simplify(),use_unicode=use_unicode)

print('Expectation value of x^2(t):')
pprint(expectation_value_x_squared_t.simplify(),use_unicode=use_unicode)

print('Expectation value of p(t=0):')
pprint(expectation_value_p.simplify(),use_unicode=use_unicode)

print('Expectation value of p(t):')
pprint(expectation_value_p_t.simplify(),use_unicode=use_unicode)

print('Expectation value of Energy(t=0):')
pprint(expectation_value_Energy.simplify(),use_unicode=use_unicode)

print('Expectation value of Energy(t):')
pprint(expectation_value_Energy_t.simplify(),use_unicode=use_unicode)