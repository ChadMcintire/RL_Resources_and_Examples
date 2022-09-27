The paper can be found at https://arxiv.org/pdf/1802.09477.pdf.

Original Pseudocode Algorithm
=============================


$\textbf{Algorithm 1}$ TD3
<ol>
  <li>Initialize critic networks $Q_{\theta_1}, Q_{\theta_2},$ and actor network
      $\pi_\phi$ with random parameters $\theta_1, \theta_2, \phi$
  <li>Initialize target networks $\theta_1\prime \leftarrow \theta_1, \theta\prime_2 \leftarrow \theta_2, \phi\prime \leftarrow \phi$
  <li>Initialize replay buffer $\mathcal{B}$
  <li>$\textbf{for}$ $t = 1$ $\textbf{to}$ $\mathcal{T}$ $\textbf{do}$
  <ol>
     <li>Select action with exploration noise $a \sim \pi_\phi(s) + \epsilon$, 
     <li>$\epsilon \sim \mathcal{N}(0, \sigma)$ and observe reward $r$ and new state s$\prime$
     <li>Store transition tuple $(s, a, r, s\prime)$ in $\mathcal{B}$
     <li>Sample mini-batch of $N$ transitions $(s, a, r, s\prime)$ from $\mathcal{B}$
     <li>$\tilde{a} \leftarrow \pi_{\phi\prime}(s\prime) + \epsilon,  \epsilon \sim clip(\mathcal{N}(0, \tilde{\sigma}), -c, c)$ <br />
     <li>$y \leftarrow r + \gamma min_{i=1,2} Q_{\theta\prime_i}(s\prime, \tilde{a})$
     <li>Update critics $\theta_i \leftarrow argmin_{\theta_i}$ $N^{-1} \sum(y - Q_{\theta_i}(s, a))^2$
     <li>$\textbf{if}$ t mod d $\textbf{then}$
     <ol>
	<li>Update $\phi$ by the deterministic policy gradient: <br />
	$\nabla_\phi J(\phi) = N^{-1} \sum \nabla_a Q_{\theta_1}(s, a) |_{a=\pi_\phi(s)} \nabla_\phi \pi_\phi(s)$
	<li>Update target networks  <br />
	$\theta_i\prime \leftarrow \tau\theta_i + (1 - tau) \theta_i\prime$ <br />
	$\phi\prime \leftarrow \tau\psi + (1 - tau) \phi_i\prime$
     <ol>	
  <ol>
<ol>
