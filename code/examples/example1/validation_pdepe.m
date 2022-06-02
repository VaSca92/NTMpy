clc;
clear all;


t = load('t.txt');
x = load('x.txt');

pe = load('phie.txt');
pl = load('phil.txt');

option = odeset('reltol', 1e-5, 'abstol', 1e-5);
tic
[sol] = pdepe(0,@pdefun,@icfun,@bcfun,x,t,option);
toc

me = sol(:,:,1);
ml = sol(:,:,2);

close all

surf(x,t,me,'Edgecolor','None','facecolor','r')
hold on
surf(x,t,pe,'Edgecolor','None','facecolor','b')
camlight headlight
lighting phong

figure()
surf(x,t,ml,'Edgecolor','None','facecolor','r')
hold on
surf(x,t,pl,'Edgecolor','None','facecolor','b')
camlight headlight
lighting phong



