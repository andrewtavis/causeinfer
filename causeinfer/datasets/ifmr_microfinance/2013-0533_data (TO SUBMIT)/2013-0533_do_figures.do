/*******************************************************************************      
Program Name: 	2013-0533_do_figures  
Contact:  		Cynthia Kinnan (c-kinnan@northwestern.edu)
Last Modified: 	5 May 2014
Purpose: 		Replicates all figures from "The miracle of microfinance? Evidence
				from a randomized evaluation" (Banerjee et al.), AEJ, 2014
Files Used: 	2013-0533_data_endlines1and2.dta
Files Created:	figure1.png
				figure2.png
				figure3.png
				figure4.png
*******************************************************************************/
version 13.1
cap log close
clear all
set more off
set mem 100m
pause on

*DATA DIRECTORY
global datadir "C:/Users/hreppst/Dropbox/Spandana/Paper/AEJ Final/Data/"

*OUTPUT DIRECTORY
global outputdir "C:/Users/hreppst/Dropbox/Spandana/Paper/AEJ Final/Data/Output"

*LOG FILE
log using 2013-0533_log_figures.smcl, replace 

cd "$outputdir"

********************************************************************************
*******        Figures 1-4: Quantile Treatment Effect Regressions       ********
********************************************************************************
use "$datadir/2013-0533_data_endlines1and2.dta", clear

*Generate business profit variables, restricting to those households that:
*		(a) Had an old business (older than 1 year) at EL1;
*		(b) Had a new business (younger than 1 year) at EL1; and
*		(c) Had a business at EL2.
gen bizprofit_1_old    = bizprofit_1 if any_old_biz==1 
gen bizprofit_1_new	   = bizprofit_1 if any_old_biz==0 & any_biz_1==1
gen bizprofit_2_biz    = bizprofit_2 if any_biz_2==1

*For each variable of interest, use bootstrapped quantile regressions to estimate
*quantile treatment effects, using intervals of .02:
program myqreg
	version 10.1
	syntax [varlist] [if] [in] [, *]
 if replay() {
		 _coef_table, `options'
 
		exit

	 }
	 qreg `0'
end

*Set seed:
set seed 65209844

foreach var of varlist informal_amt_1 bizprofit_1_old bizprofit_1_new bizprofit_2_biz {
	preserve
	
	forval i=.05(.02).97 {
		di `i'
		
		*OLS regression:
		reg `var' treatment, cluster(areaid)
		scalar betaols=_b[treatment]

		*Quantile regression:
		bs, reps(500) cluster(areaid): myqreg `var' treatment, q(`i') level(90)
		scalar beta = _b[treatment]
		scalar sigma = _se[treatment]
		
		scalar ts = invttail(e(N), 0.05)
		scalar cil = beta - (ts*sigma)
		scalar ciu = beta + (ts*sigma)
		matrix `var'_qreg = [nullmat(`var'_qreg) \ `i', betaols, beta, cil, ciu]
		}

	clear
	svmat `var'_qreg
	ren `var'_qreg1 qtile
	ren `var'_qreg2 ols_treatment
	ren `var'_qreg3 treatment
	ren `var'_qreg4 treatment_cilo
	ren `var'_qreg5 treatment_cihi
	
	tempfile `var'
	save ``var'', replace

	restore
}
	

********************************************************************************
*****       Fig. 1: Quantile Treatment Effects, Informal Borrowing        ******
********************************************************************************
use `informal_amt_1', clear

gen y0=0
gen Percentile=qtile*100
replace y0=. if Percentile<5 | Percentile>95
format treatment ols_treatment treatment_cihi treatment_cilo %6.0f 

*Truncating confidence intervals for better visualization:
replace treatment_cihi=20000 if treatment_cihi>20000
replace treatment_cihi=. if Percentile==95

label var treatment "Quantile treatment effect"
label var ols_treatment "OLS"
la var treatment_cihi "90% C.I"

twoway (line ols_treatment Percentile, lpattern(dash)) ///
	   (line treatment Percentile, lwidth(medthick) lcolor(gray lpattern(line))) ///
	   (line y0 Percentile, lcolor(black)) ///
	   (line treatment_cihi Percentile, lwidth(thin) lpattern(longdash_dot_dot) lcolor(navy)) ///
	   (line treatment_cilo Percentile, lwidth(thin) lpattern(longdash_dot_dot) lcolor(navy)), ///
	    xlabel(5 10 20 30 40 50 60 70 80 90 95) ///
		ylabel(-18000 -6000 0 6000 18000) ///
		title(Treatment effect on informal borrowing*) ///
		subtitle((endline 1)) ///
		scheme(s1manual) legend(order(1 2 4)) legend(rows(1))
graph export "figure1.png", replace


********************************************************************************
**   Fig. 2: Quantile Treatment Effects, Endline 1 Profits (Old Businesses)   **
********************************************************************************
use `bizprofit_1_old', clear

gen y0=0
gen Percentile=qtile*100
replace y0=. if Percentile<5 | Percentile>95
format treatment ols_treatment treatment_cihi treatment_cilo %6.0f 

label var treatment "Quantile treatment effect"
label var ols_treatment "OLS"
la var treatment_cihi "90% C.I"

twoway (line ols_treatment Percentile, lpattern(dash)) ///
	   (line treatment Percentile, lwidth(medthick) lcolor(gray lpattern(line))) ///
	   (line y0 Percentile, lcolor(black)) ///
	   (line treatment_cihi Percentile, lwidth(thin) lpattern(longdash_dot_dot) lcolor(navy)) ///
	   (line treatment_cilo Percentile, lwidth(thin) lpattern(longdash_dot_dot) lcolor(navy)), ///
	    xlabel(5 10 20 30 40 50 60 70 80 90 95) ///
	    ylabel(-3000(3000)9000) ///
		title(Treatment effect on business profits) ///
		subtitle((HHs who have an old business*, endline 1)) ///
		scheme(s1manual) legend(order(1 2 4)) legend(rows(1))
graph export "figure2.png", replace


********************************************************************************
**   Fig. 3: Quantile Treatment Effects, Endline 1 Profits (New Businesses)   **
********************************************************************************
use `bizprofit_1_new', clear

gen y0=0
gen Percentile=qtile*100
replace y0=. if Percentile<5 | Percentile>95
format treatment ols_treatment treatment_cihi treatment_cilo %6.0f 

label var treatment "Quantile treatment effect"
label var ols_treatment "OLS"
la var treatment_cihi "90% C.I"

*Truncating confidence intervals for better visualization:
replace treatment_cihi=12000 if treatment_cihi>12000
replace treatment_cilo=-12000 if treatment_cilo<-12000

twoway (line ols_treatment Percentile, lpattern(dash)) ///
	   (line treatment Percentile, lwidth(medthick) lcolor(gray lpattern(line))) ///
	   (line y0 Percentile, lcolor(black)) ///
	   (line treatment_cihi Percentile, lwidth(thin) lpattern(longdash_dot_dot) lcolor(navy)) ///
	   (line treatment_cilo Percentile, lwidth(thin) lpattern(longdash_dot_dot) lcolor(navy)), ///
	    xlabel(5 10 20 30 40 50 60 70 80 90 95) ///
	    ylabel(-12000(6000)12000) ///
		title(Treatment effect on business profits) ///
		subtitle((HHs who have a new business*, endline 1)) ///
		scheme(s1manual) legend(order(1 2 4)) legend(rows(1))
graph export "figure3.png", replace


********************************************************************************
**   Fig. 4: Quantile Treatment Effects, Endline 2 Profits (All Businesses)   **
********************************************************************************
use `bizprofit_2_biz', clear

gen y0=0
gen Percentile=qtile*100
replace y0=. if Percentile<5 | Percentile>95
format treatment ols_treatment treatment_cihi treatment_cilo %6.0f 

label var treatment "Quantile treatment effect"
label var ols_treatment "OLS"
la var treatment_cihi "90% C.I"

*Truncating confidence intervals for better visualization:
replace treatment_cihi=. if treatment_cihi>6000

twoway (line ols_treatment Percentile, lpattern(dash)) ///
	   (line treatment Percentile, lwidth(medthick) lcolor(gray lpattern(line))) ///
	   (line y0 Percentile, lcolor(black)) ///
	   (line treatment_cihi Percentile, lwidth(thin) lpattern(longdash_dot_dot) lcolor(navy)) ///
	   (line treatment_cilo Percentile, lwidth(thin) lpattern(longdash_dot_dot) lcolor(navy)), ///
	    xlabel(5 10 20 30 40 50 60 70 80 90 95) ///
	    ylabel(-1500(1500)6000) ///
		title(Treatment effect on business profits) ///
		subtitle((full sample of business owners, endline 2)) ///
		scheme(s1manual) legend(order(1 2 4)) legend(rows(1))
graph export "figure4.png", replace


log close
