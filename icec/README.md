`icec` calculates the asymptotic cross section for atom-atom systems, neglecting interatomic vibrational modes.

The cross section is calculates as

$$\sigma (k) = 
     \frac{3 \hbar^4 c^2}{8 \pi m_e} 
     \frac{g_{A^-}}{g_A} 
     \frac{\sigma_\text{PI}^{A^-}(\epsilon_k) \sigma_\text{PI}^B(\omega)}
         {R^6 \epsilon_k(\hbar\omega)^2}
$$

where the transferred energy $\hbar\omega$ is

$$\hbar\omega = \epsilon_k + IP_{A}$$

It can also calculate the overlap contribution.

You can either caculate the cross section for a given electron energy or for a range of energies.

`interICEC` includes the interatomic vibrational modes, but so far only bound states.

The Morse Potential is given by

$$V(R) = D_e \left( 1 - e^{\alpha (R-R_\text{eq})}\right)^2 - D_e$$

With the width parameter as

$$\alpha = \omega_0 \sqrt{\frac{\mu}{2 D_e}}$$

Further parameters are 

$$\lambda = \frac{\sqrt{2 m D_e}}{\alpha \hbar}$$

The energy levels for the bound states are given by

$$E_v = w_0 \left(v + \frac{1}{2}\right) - \frac{w_0^2}{4 D_e} \left(v + \frac{1}{2}\right)^2$$

$$v \in ( n \in \mathbb{N}_0 |  n < \lambda - \frac{1}{2} ) $$

The cross section is calculated as

$$\sigma (k, i \to f) 
     = \frac{3 \hbar^4 c^2 }{8 \pi m_e} 
     \frac{g_{A^-}}{g_A} 
     \frac{\sigma_\text{PI}^A(E) \sigma_\text{PI}^B(E - \Delta E)}{\epsilon_k E(E - \Delta E)} 
     |\langle \chi_{AB^+}| R^{-3}|\chi_{A^+B}\rangle 
     |^2$$

 
$$ \begin{aligned}
     \sigma (k, i \to f) 
     &= \frac{3 (\hbar c)^4 }{4 \pi} 
     \frac{\sigma_\text{PR}^A(\epsilon_k) \sigma_\text{PI}^B(E - \Delta E)}
         {E^3(E - \Delta E)} 
     \left|    
     \bra{\chi_{AB^+}} R^{-3} \ket{\chi_{A^+B}} 
     \right|^2 \\
     &= \frac{3 \hbar^4 c^2 }{8 \pi m_e} 
     \frac{g_{A^-}}{g_A} 
     \frac{\sigma_\text{PI}^A(E)  \sigma_\text{PI}^B(E - \Delta E)}
         {\epsilon_k E(E - \Delta E)} 
     \left|    
     \bra{\chi_{AB^+}} R^{-3} \ket{\chi_{A^+B}} 
     \right|^2
 \end{aligned} $$
     
where the transferred energy is given as

$$E = \epsilon_k + IP_{A}$$

and the energy that goes into the vibrational motion is

$$ \Delta E = E^{AB^+}\_{vib,f} - E^{A^+B}_{vib,i} $$

The threshhold energy is further given as

$$\begin{aligned}
\epsilon_t &= IP_A - IP_B - \Delta E\\
&= IP_A - IP_B + E^{A^+B}\_{\text{vib},i} - E^{AB^+}_{\text{vib},f} 
\end{aligned} $$

It can also calculate the overlap contribution.

You can either caculate the cross section for a given electron energy or for a range of energies.
