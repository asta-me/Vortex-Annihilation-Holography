Vol. 50, No. 22/15 November 2025/ _Optics Letters_ 

**6927** 





## **Vortex annihilation to reduce phase singularities and resultant speckles in computer-generated holography** 

**Xiaomeng Sui,**<sup>**1,2**</sup> **Zehao He,**<sup>**1**</sup> **Daping Chu,**<sup>**2,3,4**</sup> **AND Liangcai Cao**<sup>**1,**</sup> ***** 

_1Department of Precision Instruments, Tsinghua University, Beijing 100084, China_ 

_2Centre for Photonic Devices and Sensors, University of Cambridge, 9 JJ Thomson Avenue, Cambridge CB3 0FA, UK_ 

_3Cambridge University–Nanjing Centre of Technology and Innovation, 21A Rongyue Road, Jiangbei New Area, Nanjing, China 4 dpc31@cam.ac.uk_ 

_*clc@tsinghua.edu.cn_ 

_Received 10 July 2025; revised 20 September 2025; accepted 6 October 2025; posted 7 October 2025; published 4 November 2025_ 

**distribution Scattering effects caused by the random phase in computer-generated holography (CGH) produce a natural depth sensation, but they also result in speckles. These speckles are linked to the optical vortices existing in the object phase and become intractable under coherent illumination. Here, we propose a vortex annihilation strategy cooperating with optimization algorithms to address phase singularities in CGH. It works out the scalar potential and vector potential in the object phase, based on which the irrotational part of the phase is extracted through an annihilation between optical vortices with opposite handedness. Our experiment validates the effectiveness of the irrotational phase for a speckle-suppressed viewing experience in holographic display, with the number of phase singularity reduced by a factor of** ∼ **100 for general grayscale objects.** © 2025 Optica Publishing Group. All rights, including for text and data mining (TDM), Artificial Intelligence (AI) training, and similar technologies, are reserved. 

https://doi.org/10.1364/OL.573204 

Computer-generated holography (CGH) is a powerful technique that enables the volumetric shaping of coherent waves to present a three-dimensional (3D) scene with depth perception [1]. Since most materials encountered in the real world are rough on the scale of an optical wavelength, various microscopic facets of a scattering surface contribute random phase distributions to the elementary wavelets [2]. Hence, an object wave in CGH can be discretely defined with the target intensity and a random phase [3], which causes speckles under coherent illumination. These speckles are intertwined by optical vortices and contain precise intensity zeros where the phase is a singularity [4], hindering perfect reconstructions of holograms. 

Various computational strategies have been proposed to suppress speckles caused by random phase. Optimization algorithms and neural networks are used to restrict the scattering caused by the random phase to suppress speckles, which is achieved by feeding the object intensity [5,6] and the bandwidth limitation [7,8] as constraints. Some can produce remarkable 

results by incorporating optical systems into the forward modal [9]. However, it has been discovered that optimization algorithms cannot inherently address phase singularities [10] and exhibit limitations in speckle suppression. Some less-random object phases are used to avoid excessive phase singularities. The quadratic phase enables speckle-free reconstructions; however, the intensity pattern associated with the quadratic phase cannot maintain its feature size during light propagation. The constant phase [11,12] and some phase formats limited in randomness [13] can produce highly photorealistic reconstructions. However, these slow-varying phases generate diffractive patterns along the direction of propagation, rather than defocus blurs, which weakens the depth sensation of 3D scenes provided by random scattering. 

To retain the use of random phase in CGH optimization, we introduce an approach that addresses phase singularities and their associated speckles through a vortex annihilation process. This is achieved by introducing an interpretation of electromagnetic fields with current flows into holography: the gradient of the object phase is interpreted as the sum of the curl of a vector potential plus the gradient of a scalar potential, and hence, the object phase is decomposed into a vortex part and an irrotational part. Vortex annihilation extracts the irrotational part of the object phase by adding reverse-spinning vortices in the CGH optimization loop, preserving partial phase randomness and a radian range of 2𝜋. The optimization synthesizes phase-only holograms (POHs), indicating that the number of remaining singularities is reduced by more than 98.81% with vortex annihilation. This approach is validated through optical reconstructions, where holograms are uploaded on a spatial light modulator (SLM) under laser illumination and experimentally present a colored multi-layer object. 

Consider a continuous object field ℎ( **r** ) = 𝑎( **r** ) exp [𝑖𝜑( **r** )], where 𝑎( **r** ) is the amplitude, 𝜑( **r** ) is the phase, and **r** = (𝑥, 𝑦) is the position vector. Analytically expanding ℎ( **r** ) into a Taylor series at an arbitrary point **r** 0 = (𝑥0, 𝑦0) in this field and taking the first-order approximation, we then have: 



0146-9592/25/000001-04 Journal © 2025 Optica Publishing Group 





<!-- Start of picture text -->
(a) (c) |<br>|<br>: ; . 7<br>:<br>ic;<br>(b|<br>:<br>:<br>at | _ = Z<br>Q Cc’ é<br><!-- End of picture text -->





<!-- Start of picture text -->
I Contour integral Pp<br>° we; | Steed S SSMQ=-1IIROSILEeo oI 1122pe repotentialVector: f ra|<br>— Vo ete sos de |<br>> g + |<br>sivdiyTaste TS tsss OW a Gesee ~——aye | 2<br>l LN [itsscreeetssss siz 0 : °<br>-1 0 ya eee potential Sy a”<br><!-- End of picture text -->



<!-- Start of picture text -->
(a) (b) Optimization With VA<br>0.5 s-----—% WithOptimizationSGDvortex (Adam)annihilation 3BSEEsis,2 aE Dea<br>i %  Quasi-Newton (L-bfgs) 7 wee i Ce<br>‘ i £ ie<br>FaoO NY R BE SLECeie BYES eeei]<br>\\ i \ | \ mt a 2 aye |<br>; —_-—_}—_ '<br>0 cy 1<br>50 100 150 200 250 300 & |<br>Number of updates E he Le 7<br><!-- End of picture text -->









<!-- Start of picture text -->
(a) Beamispliter<br>Doublet Lz<br>Doublet Iris Polarizer<br>L3 s— Doublet L;<br>—— 8 38<br>vy<br>on” ff<br>N ho gin == Multiwavelength<br>8 (an Laser<br>30°<br>(b) (9)<br>Nz = 433194 Ng = 423556 Ns=3 Ns =5026<br>o = 0.0476 o = 0.0688 o = 0.0131 o = 0.0410<br>a40 ’ 4 he.8 10MLat 0.2 0.4mm a4}0 4 ; LL.8mmSLat0 0.27 0.4mm<br><!-- End of picture text -->



<!-- Start of picture text -->
(a) 170 mm — (9) 170 mm & e<br>> ee —— Ys yr:<br>=“ > Bt ia =! t<br>ou<br>230 mm | 230 mm |<br>on an’.<br>ae — 3 =<br><!-- End of picture text -->

# ~~ee~~ 

