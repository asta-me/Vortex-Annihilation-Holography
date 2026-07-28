

<!-- Start of picture text -->
<<“= = =<br>Optics Letters<br><!-- End of picture text -->

# **Vortex annihilation to reduce phase singularities and resultant speckles in computer-generated holography: supplemental document** 

This document provides supplementary information to "Vortex annihilation to reduce phase singularities and resultant speckles in computer-generated holography". Included are additional derivations, details about the algorithm, and experimental implementations. 

## **1. ELLIPTICAL MAPPING UNDER THE FIRST PERTURBATION** 

This section aims to show how Eqs. (2) and (3) in the Letter are obtained through the first-order Taylor series. A scalar wave with speckles reconstructed from a computer-generated hologram can be represented by _h_ ( **r** ) = _a_ ( **r** ) exp[ _iφ_ ( **r** )], where _a_ ( **r** ) is the amplitude, _φ_ ( **r** ) is the phase, and **r** = ( _x_ , _y_ ) is the position vector. Given that we can also put the complex field in the form of real and imaginary parts _h_ ( **r** ) = _ξ_ ( **r** ) + _iη_ ( **r** ), the Taylor expansion at the point **r** 0 = ( _x_ 0, _y_ 0) can be expressed as: 



where _ξx_<sup>_′_</sup> , _ηx_<sup>_′_</sup> , _ξy_<sup>_′_</sup> and _ηy_<sup>_′_</sup> are the first-order partial derivatives at **r** 0, and _o_ ( **r** ) is the remainder of the series. Consider a sequence of points on a closed circular contour _C_ centered on **r** 0, as depicted in Fig. 1(a). Establish polar coordinates based on: _x_ = _x_ 0 + _ρ_ cos _θ_ and _y_ = _y_ 0 + _ρ_ sin _θ_ , where _ρ_ is the radius and _θ_ is the angle that describes _C_ in a positive sense of rotation (anticlockwise) as _θ_ goes from 0 to 2 _π_ . Taking the first-order approximation, we have the real and imaginary parts of Eq. (S1) recast as 



where _ξ_ 0 = _ξ_ ( **r** 0) and _η_ 0 = _η_ ( **r** 0). Introducing intermediate quantities defined as: 









Then the real and imaginary parts of _h_ ( _ρ_ , _θ_ ) can be rewritten as: 



This pair of parametric equations describes an enclosed ellipse contour _C_<sup>_′_</sup> centered at ( _ξ_ 0, _η_ 0), since ( _ξ_ , _η_ ) can always trace back to its initial value when _θ_ goes from 0 to 2 _π_ . This mapping was well demonstrated by Fried et al. in their study of phase singularities in an optical wave propagated under atmospheric turbulence [1]. 

## **2. SINGULARITY IDENTIFICATION** 

This section aims to present how phase singularities are identified through Eq. (4) and in practical Eq. (7) in the Letter. Assuming that the phase distribution _φ_ has only one positive singularity, as shown in Fig. S1(a), it has a topological index _Q_ = 1. The gradient of this phase distribution _∇φ_ is a vector field with only _x_ and _y_ components, whose orientations are depicted in Fig. S1(b). 



<!-- Start of picture text -->
(a) 5.0. Vortex phase (Q = 1) 3 (b) ioan PhaseceeeeneaneeeetNygradient (Q = 1) (c) Integration of curl (Q = 1)<br>25 : PE eeeTSE ots<br>&> 00 0 28 BOOT> VANANA ASSES aeTT 2 |po eé<br>MR 4<br>-5.0 3 -5.0 NAA ANS Steere eee eee 4 o>2<br>5.0 -25 00 25 5.0 5.0 -25 00 25 5.0 -2 4 -2<br>x (a.u.) x (a.u.) ay) 2 4 -4 4<br><!-- End of picture text -->



<!-- Start of picture text -->
(a) s++---- Optimization (b) s++---- Optimization (c) Optimization With VA<br>— With vortex annihilation —— With vortex annihilation é ze<br>15 : ; 05 : ; - A NS<br>%* SGD (Adam) ¢ %* SGD (Adam) & } é<br>. %* Quasi-Newton (L-bfgs) 0.4 %* ~Quasi-Newton (L-bfgs) a | KS ae<br>,\ * GS \s * GS H<br>ed? z tA. | H é7 i<br>uf 0.24 = h 3 ae on au . |<br>0.4 NI AY i =<br>50 Number100 150of updates200 250 300 50 Number100 150of updates200 250 300 asa |Ewe<br><!-- End of picture text -->

A A 



<!-- Start of picture text -->
Hn (r,z)= V Lov} (r,z) exp lig (r,z) ]<br>PE LH (r,2)=a(r,z)exp lip” (r.2)]<br>go? (+1) (r,z)==a9 (r,2z) a® (1,2)f = Ion (FZ)<br>Target intensity<br>Vortex annihilation<br>9.(r,2)= 9 (r.2)— 9(r,2)] |6 (w) = ang [H (u)] fpi+<br>ee detection; T(Bandwidthheh)<br>aEbvows)Or dr limitation<br>r=(x,y)<br>; a Constraint H :<br>}—> Vortex annihilation — d(u)= 4° (u)<br><!-- End of picture text -->



<!-- Start of picture text -->
4<br>6<br>5<br>2 4<br>a<br>O3<br>2<br>1<br>0<br>400 450 500 550 600 650 700<br>Wavelength(nm)<br><!-- End of picture text -->

3. X. Sui, Z. He, D. Chu, and L. Cao, “Non-convex optimization for inverse problem solving in computer-generated holography,” Light. Sci. & Appl. **13** , 158 (2024). 

5 

