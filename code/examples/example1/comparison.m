clc;
clear all;


t = load( 'time.txt');
x = load('space.txt');

Te_py = load('temp_elec.txt');
Tl_py = load('temp_latt.txt');

option = odeset('reltol', 1e-6, 'abstol', 1e-6);

tic
[sol] = pdepe(0,@fun_PDE,@fun_IC,@fun_BC,x,t,option);
toc

Te_m = sol(:,:,1);
Tl_m = sol(:,:,2);

close all

surf( x, t, Te_m ,'Edgecolor', 'None', 'facecolor', 'r')
hold on
surf( x, t, Te_py, 'Edgecolor', 'None', 'facecolor', 'b')
camlight headlight
lighting phong

figure()
surf( x, t, Tl_m , 'Edgecolor', 'None', 'facecolor', 'r')
hold on
surf( x, t, Tl_py, 'Edgecolor', 'None', 'facecolor', 'b')
camlight headlight
lighting phong



