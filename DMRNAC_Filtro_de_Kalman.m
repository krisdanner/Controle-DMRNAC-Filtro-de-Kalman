% Controle ROBUSTO NEURO-ADAPTATIVO MRAC com Filtro de Kalman
% Tipo: Direto 
% Regra: Lyapunov
% Método: Modelo de referência
% Modificação sigma + Recuperação de desempenho + Filtro de Kalman
%------------------------------------------------------------------
% Autor: Christian Danner Ramos de Carvalho
%------------------------------------------------------------------

clear
clc

rng(8);

% ----- Sistema em Espaço de Estados
A = [0 1;
    0 -0.1190];
B = [0;
    373.3151];
C = [1 0];
D = 0;

sys = ss(A,B,C,D);
display(sys);

% Condições iniciais
x = [0; 0];
xm = [0; 0];
xmi = [0; 0];
Psi = [0; 0]; % Vetor de estado do governador de comando

% erros do sistema rastreado
sigma_w = 0.01; % desvio padrão de S (m)
sigma_s = 0.02; % desvio padrão de V (m/s)

% matriz de covariâncias do sistema 
Q_K = [sigma_w^2 0;0 sigma_s^2];

% erros do sensor
sigma_v = 0.1; % desvio padrão do sensor do estado S (m)

% matriz de covariâncias do sensor 
R_K = sigma_v^2;

% primeira leitura do sensor
y = C*x + sqrt(R_K)*randn;
Y = y; % armazenamento do sensor para plot

% estado inicial da esperança do sistema ("xK" para esperança de "x", e "z" para esperança de "y")
xK = [2;0];
XK = xK; % armazenamento dos estados da esperança para plot

% covariância inicial da esperança do erro e(k)
P_K = Q_K; % sempre existe incerteza no valor inicial

% ganho de Kalman inicial
K = P_K*C'*(C*P_K*C' + R_K)^-1;
GK = K; % armazenamento do ganho de Kalman para plot

% leitura inicial da esperança do sensor
z = C*xK;
Z = z; %armazenamento dos estados da esperança do sensor para plot
e = abs(x(1,1)-z); % erro entre o estado real e sua esperança

% Controlador Nominal
% Q_lqr = [1 0;   
%      0 .1];
% R_lqr = .1;
% K1 = lqr(A,B,Q_lqr,R_lqr);
K1 = acker(A,B,[-1 -2]); % Ganho de realimentação de estados
K2 = -inv(C*inv(A-B*K1)*B); % Ganho de alimentação direta da referência

% Taxa de aprendizagem
gamma = 3; %2

% Modificação sigma
sigma = 500; %500

% Recuperação de desempenho
lambda = 20; %0.1

% Condições de Correspondência do Modelo
Am = A-B*K1;
Bm = B*K2;

% Sinal de correção (Recuperação de Desempenho)
G = inv(K2)*inv(B'*B)*B'; % Matriz do sinal governador de comando
Omega = B*inv(B'*B)*B';

% Equação algébrica de Lyapunov
Q = eye(2);
P = lyap(Am',Q); %solução da equação algébrica de Lyapunov

% Condições iniciais para W_hat
n = 500; % Quantidade de neurônios para cada estado do sistema
b = 100; % limite do domínio D (1000 metros de altitude)
W_hat = zeros(2*n+1,1);

% Parâmetros da RNA
centers = linspace(-b, b, n); % Vetor de centros dos neurônios
Theta = zeros(2*n, 1); % Inicialização do vetor regressor da RNA

% Simulação
ft = 100;
dt = 0.001;
index = 1;

for k = 0:dt:ft
    
    % delta = 0; 
    delta = 1+x(1)+x(2)+x(1)^2+sin(x(1))+cos(x(1))+sin(x(2))+cos(x(2)); 
    % delta = 1+x(1)^2+sin(x(1))*x(1)^3+cos(x(1))*x(1)^4+x(2)*x(1)+x(2)^2+abs(x(1))^2+x(2)*abs(x(1))^5; % Incerteza

    % Referência
    % if k <= 10
    %     r = 1;
    % end
    % if k > 10
    %     r = -1;
    % end
    if k < 2
        r = 0;
    end
    if k >= 2
        r = 2;
    end
    if k >= 6
        r = -2;
    end
    if k >= 10
        r = 2;
    end
    if k >= 14
        r = -2;
    end

    %%%%%%%%%%%%%%%%%%%

    if k >= 18
        r = 2;
    end
    if k >= 22
        r = -2;
    end
    if k >= 26
        r = 2;
    end
    if k >= 30
        r = -2;
    end
    if k >= 34
        r = 2;
    end
    if k >= 38
        r = -2;
    end
    if k >= 42
        r = 2;
    end
    if k >= 46
        r = -2;
    end

    % if k <= 10
    %     r = 1;
    % end
    % if k > 10
    %     r = 2;
    % end
    % if k > 20
    %     r = 3;
    % end
    % if k > 30
    %     r = 4;
    % end
    % if k > 40
    %     r = 5;
    % end
    % if k > 50
    %     r = 10;
    % end
    % if k > 60
    %     r = 10;
    % end
    % if k > 61
    %     r = -10;
    % end
    % if k > 62
    %     r = 10;
    % end
    % if k > 63
    %     r = 20;
    % end
    % if k > 64
    %     r = 30;
    % end
    % if k > 65
    %     r = 40;
    % end
    
    % Montagem do vetor regressor da RNA
    for i = 1:n
        Theta(i) = exp(-0.25*(abs(xK(1)-centers(i)))^2); % RBFs
        Theta(i+n) = exp(-0.25*(abs(xK(2)-centers(i)))^2);
    end
    Theta(2*n+1) = 1; % Bias
    
    % Sinal de correção para recuperação
    Psi = Psi + dt*(-lambda*(Psi-(xK - xm))); % Dinâmica do governador de comando
    v = lambda*Psi + (Am-lambda*eye(2))*((xK-xm)); % Sinal do governador de comando
    
    % Sinal de controle
    u = -K1*xK + K2*(r+G*v) - W_hat'*Theta;
    xm = xm + dt*(Am*xm + Bm*r + Omega*v); % modelo de referência
    xmi = xmi + dt * (Am*xmi + Bm*r); % apenas para plotagem do modelo ideal
    W_hat = W_hat + dt*(gamma*(Theta*(xK-xm)'*P*B - sigma*W_hat)); % Atualização dos pesos (adaptação)
    
    % Sistema Atual
    x = x + dt*(A*x + B*(u + delta) + sqrt(Q_K)*randn(size(x)));

    % leitura do sensor real
    y = C*x + sqrt(R_K)*randn;

    % predição dos estados por meio da esperança
    xK = xK + dt*(A*xK + B*(u + delta));

    % cálculo da covariância da esperança
    P_K = A*P_K*A' + Q_K;

    % esperança do sensor
    z = C*xK;
    e = abs(x(1,1)-z); % erro entre o estado real e sua esperança

    % GANHO DE KALMAN
    K = P_K*C'*(C*P_K*C' + R_K)^-1;

    % correção da esperança do sistema
    xK = xK + K*(y-z);

    % correção da covariância
    P_K = (eye(size(Q_K)) - K*C)*P_K;
   
    % Gravação dos dados
    delta_rec(index,1) = delta;
    w_theta_rec(index,1) = W_hat'*Theta;
    r_rec(index,1) = r;
    xm_rec(index,1:2) = xm;
    xmi_rec(index, 1:2) = xmi;
    x_rec(index,1:2) = x;
    y_rec(index,1) = y;
    xK_rec(index,1:2) = xK;
    K_rec(index,1:2) = K;
    eK_rec(index,1) = e;
    u_rec(index,1) = u;
    t_rec(index,1) = k;
    e_rec(index,1:2) = x-xmi; 
    e_rbf_rec(index,1) = delta - W_hat'*Theta;
    index = index + 1;
end

figure; 
hold on; box on; grid;
%title('MRAC - Kalman Filter','fontsize',16,'interpreter','latex');
p1 = plot(t_rec,r_rec,'r:');
set(p1,'linewidth',2);
p2 = plot(t_rec,y_rec(:,1),'g-');
set(p2,'linewidth',.5);
p3 = plot(t_rec,xmi_rec(:,1),'b--');
set(p3,'linewidth',3);
p4 = plot(t_rec,x_rec(:,1),'k-');
set(p4,'linewidth',2);
p5 = plot(t_rec,xK_rec(:,1),'m-');
set(p5,'linewidth',1);
xlabel('$t$ (s)','fontsize',16,'interpreter','latex');
ylabel('$x_1$ (m)','fontsize',16,'interpreter','latex');
legend('r','Sensor','$x_{1}$m','$x_1$','$x_1$ Kalman','interpreter','latex');
axis tight;

% Plot

figure;
subplot(4,1,1); hold on; box on; grid;
%title('MRAC - Kalman Filter','fontsize',16,'interpreter','latex');
p1 = plot(t_rec,r_rec,'r:');
set(p1,'linewidth',2);
p2 = plot(t_rec,y_rec(:,1),'g-');
set(p2,'linewidth',.5);
p3 = plot(t_rec,xmi_rec(:,1),'b--');
set(p3,'linewidth',3);
p4 = plot(t_rec,x_rec(:,1),'k-');
set(p4,'linewidth',2);
p5 = plot(t_rec,xK_rec(:,1),'m-');
set(p5,'linewidth',1);
xlabel('$t$ (s)','fontsize',16,'interpreter','latex');
ylabel('$x_1$ (m)','fontsize',16,'interpreter','latex');
legend('r','Sensor','$x_{1}$m','$x_1$','$x_1$ Kalman','interpreter','latex');
axis tight;

% figure;
% subplot(4,1,1); hold on; box on; grid;
% %title('DMNAC, mod-$\sigma$, sinal GC','fontsize',16,'interpreter','latex');
% p0 = plot(t_rec,r_rec,'r:');
% set(p0,'linewidth',4);
% p1 = plot(t_rec, xmi_rec(:,1), 'c-');
% set(p1, 'linewidth', 3);
% % p1 = plot(t_rec,xm_rec(:,1),'b--');
% % set(p1,'linewidth',3);
% p2 = plot(t_rec,x_rec(:,1),'k-');
% set(p2,'linewidth',2);
% xlabel('$t$ (s)','fontsize',10,'interpreter','latex');
% ylabel('$x_1$ (m)','fontsize',16,'interpreter','latex');
% legend('$r$','$x(1)_m$','x(1)','fontsize',10,'interpreter','latex');
% axis tight;

subplot(4,1,2); hold on; box on; grid;
p1 = plot(t_rec,xmi_rec(:,2),'c-');
set(p1,'linewidth',3);
p2 = plot(t_rec,x_rec(:,2),'k-');
set(p2,'linewidth',2);
xlabel('$t$ (s)','fontsize',10,'interpreter','latex');
ylabel('$x_2$ (m/s)','fontsize',16,'interpreter','latex');
legend('$x(2)_m$','$x(2)$','fontsize',10,'interpreter','latex');
axis tight;

subplot(4,1,3); hold on; box on; grid;
p3 = plot(t_rec,u_rec,'b-');
set(p3,'linewidth',3);
xlabel('$t$ (s)','fontsize',10,'interpreter','latex');
ylabel('$u(t)$','fontsize',16,'interpreter','latex');
legend('$u(t)$','fontsize',10,'interpreter','latex');
axis tight;

subplot(4,1,4); hold on; box on; grid;
plot(t_rec,delta_rec,'g-','linewidth',3); hold on; box on; grid;
plot(t_rec,w_theta_rec,'r--','linewidth',3); grid;
xlabel('$t$ (s)','fontsize',10,'interpreter','latex');
ylabel('$\Delta(x)$','fontsize',16,'interpreter','latex');
legend('$\Delta(x)$','$\hat{W}^T\Theta(x)$','fontsize',10,'interpreter','latex');
%axis tight;