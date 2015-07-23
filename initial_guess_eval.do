/* initial_guess_eval.do */
*------------------------------------------------------------------*
/* This program evaluates model convergence as a function of initial guesses
of r and w to solve for the SS eq'm */
*------------------------------------------------------------------*;




#delimit ;
set more off;
capture clear all;
capture log close;
set memory 8000m;
cd "~/repos/firm_sandbox/" ;
log using "~/repos/firm_sandbox/initial_guess_eval.log", replace ;


local datapath "~/repos/firm_sandbox/" ;
local graphpath "~/repos/firm_sandbox/Graphs/" ;

set matsize 800 ;

/* read in data from SS_v2pt1_mktclear.py */
insheet using "`datapath'/init_guess_output.csv", comma clear ;

/* rename variables */
replace v11 = "" if v11 == "nan" ;
destring v11, replace ;
rename v1 r_guess ;
rename v2 w_guess ;
rename v3 rss ;
rename v4 wss ;
rename v5 cap_diff ;
rename v6 labor_diff ;
rename v7 RC1 ;
rename v8 RC2 ;
rename v9 RC1_2 ;
rename v10 RC2_2 ;
rename v11 euler_error ;
gen abs_labor_diff = abs(labor_diff) ;
gen abs_cap_diff = abs(cap_diff) ;
 
save "`datapath'/init_guess_output.dta", replace ;

twoway (connected cap_diff r_guess if w_guess == 1, msize(small) mcolor(blue) lcolor(blue) msymbol(D))
(connected labor_diff r_guess, msize(small) mcolor(red) lcolor(red) msymbol(O)),
	title("Market clearing diffs by r_guess") 
	xtitle("Initial guess of r") 
	ytitle("Diff in market clearing condition")
	/*xscale(range(1 8)) 
	xlabel(1(1)8) 
	ylabel(0(10)80,grid) */
	legend(label(1 "Capital Market")) 
	legend(label(2 "Labor Market"))  
	scheme(s1mono) 
	saving(graph1, replace); 
graph export "`graphpath'/market_clearing_diffs.pdf", replace;

twoway (connected cap_diff r_guess if w_guess == 1, msize(small) mcolor(blue) lcolor(blue) msymbol(D)),
	title("Market clearing diffs by r_guess") 
	xtitle("Initial guess of r") 
	ytitle("Diff in market clearing condition")
	/*xscale(range(1 8)) 
	xlabel(1(1)8) 
	ylabel(0(10)80,grid) */
	legend(label(1 "Capital Market")) 
	scheme(s1mono) 
	saving(graph1, replace); 
graph export "`graphpath'/cap_market_clearing_diffs.pdf", replace;

twoway(connected labor_diff r_guess if w_guess == 1, msize(small) mcolor(red) lcolor(red) msymbol(O)),
	title("Market clearing diffs by r_guess") 
	xtitle("Initial guess of r") 
	ytitle("Diff in market clearing condition")
	/*xscale(range(1 8)) 
	xlabel(1(1)8) 
	ylabel(0(10)80,grid) */
	legend(label(1 "Labor Market"))  
	scheme(s1mono) 
	saving(graph1, replace); 
graph export "`graphpath'/labor_market_clearing_diffs.pdf", replace;


twoway (connected cap_diff r_guess if r_guess > 0.4 & r_guess < 0.75 & w_guess == 1, msize(small) mcolor(blue) lcolor(blue) msymbol(D)),
	title("Market clearing diffs by r_guess") 
	xtitle("Initial guess of r") 
	ytitle("Diff in market clearing condition")
	/*xscale(range(1 8)) 
	xlabel(1(1)8) 
	ylabel(0(10)80,grid) */
	legend(label(1 "Capital Market")) 
	scheme(s1mono) 
	saving(graph1, replace); 
graph export "`graphpath'/cap_market_clearing_diffs_zoomed.pdf", replace;

twoway (connected cap_diff r_guess if r_guess > 0.4 & r_guess < 0.75 & cap_diff > -1e-2 & cap_diff < 1e-2 & w_guess == 1, msize(small) mcolor(blue) lcolor(blue) msymbol(D)),
	title("Market clearing diffs by r_guess") 
	xtitle("Initial guess of r") 
	ytitle("Diff in market clearing condition")
	/*xscale(range(1 8)) 
	xlabel(1(1)8) 
	ylabel(0(10)80,grid) */
	legend(label(1 "Capital Market")) 
	scheme(s1mono) 
	saving(graph1, replace); 
graph export "`graphpath'/cap_market_clearing_diffs_zoomed2.pdf", replace;

twoway (connected cap_diff r_guess if r_guess > 0.4 & r_guess < 0.75 & w_guess == 1, msize(small) mcolor(blue) lcolor(blue) msymbol(D)),
	title("Market clearing diffs by r_guess") 
	xtitle("Initial guess of r") 
	ytitle("Diff in market clearing condition")
	/*xscale(range(1 8)) 
	xlabel(1(1)8) 
	ylabel(0(10)80,grid) */
	legend(label(1 "Capital Market")) 
	scheme(s1mono) 
	saving(graph1, replace); 
graph export "`graphpath'/cap_market_clearing_diffs_zoomed_wage1.pdf", replace;


fucking stop ;

twoway (connected cap_diff r_guess, msize(small) mcolor(blue) lcolor(blue) msymbol(D))
(connected lab_diff r_guess, msize(small) mcolor(red) lcolor(red) msymbol(O))
(connected price year if vehicle_id == 11, msize(small) mcolor(orange) lcolor(organge) msymbol(S))
(connected price year if vehicle_id == 12, msize(small) mcolor(green) lcolor(green) msymbol(T))
(connected price year if vehicle_id == 5, msize(small) mcolor(black) lcolor(black) msymbol(T)),
	title("Average vehicle price by model year, cars most interested in") 
	xtitle("Model Year") 
	ytitle("Mean Price")
	/*xscale(range(1 8)) 
	xlabel(1(1)8) */
	ylabel(0(10)80,grid)
	legend(label(1 "Toyota Tundra")) 
	legend(label(2 "Toyota Sequoia")) 
	legend(label(3 "GMC Sierra 2500, Diesel")) 
	legend(label(4 "Chevy Silverado 2500, Diesel"))  
	legend(label(5 "Chevy Suburban"))  
	scheme(s1mono) 
	saving(graph1, replace); 
graph export "`graphpath'/top_cars_by_year.pdf", replace;


capture log close ;
