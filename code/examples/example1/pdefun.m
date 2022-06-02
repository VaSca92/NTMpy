function [c,f,s] = pdefun(x,t,u,dudx)
    c1e = 0.112.*u(1)*6500;
    c2e = 0.025.*u(1)*5100;
    c1l = 450*6500;
    c2l = 730*5100;
    
    c = [c1e*(x<40e-9) + c2e*(x>=40e-9); c1l*(x<40e-9) + c2l*(x>=40e-9)];
    
    f1 = 6*dudx(1); f2 =  12*dudx(1);
    f = [f1*(x<40e-9) + f2*(x>=40e-9); dudx(2)];
    
    lamb = 45*1e-9; J = 2e18; t0 = 2e-12; sig = 2e-12;
    G = 5e17;
    
    s = [J*exp(-x/(lamb))*exp(-(t-t0)^2/2/sig^2) + G*(u(2)-u(1)); G*(u(1)-u(2))]; 
end





 


