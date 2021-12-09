clear all;

init_extensor_muscle;
init_flexor_muscle;
tsim = 5;
release_time = 0.15;

sim('throwing_arm');