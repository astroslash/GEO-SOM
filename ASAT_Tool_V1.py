# -*- coding: utf-8 -*-
"""
Created on Fri May 20 11:59:42 2022

@author: kenny
"""
###############################################################################
######################## Import Packages ######################################
#Import GUI Packages
import tkinter as tk
from tkinter import ttk

#Import Base Packages
import numpy as np
import numpy.linalg as LA
import io
import json
import xml.etree.ElementTree as ET
import warnings

#Load astropy packages
from astropy import coordinates as coord
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import TEME, GCRS


#Load poliastro packages
from poliastro.twobody import Orbit
from poliastro.core.iod import vallado
from poliastro.bodies import Earth
from poliastro.util import time_range
from poliastro.ephem import Ephem
from poliastro.frames import Planes
from poliastro.maneuver import Maneuver
from poliastro.plotting import StaticOrbitPlotter
#Load modules for plotting the groundtrack

import astroquery
import numpy.lib.recfunctions

#load matlibplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#load packages to read from 18 SDS satcat through celestrak
import httpx
from sgp4 import exporter, omm
from sgp4.api import Satrec

#declare global variables
def main():
    ###############################################################################
    ###################### Define Calculation Functions ###########################
    #Define functions for the proceeding calculations
    # These are the functions necessary for reading TLEs from Celestrak
    # From https://github.com/poliastro/poliastro/blob/main/contrib/satgpio.py
    def _generate_url(catalog_number, international_designator, name):
        params = {"CATNR": catalog_number, "INTDES": international_designator, "NAME": name}
        param_names = [
            param_name
            for param_name, param_value in params.items()
            if param_value is not None
        ]
        if len(param_names) != 1:
            raise ValueError(
                "Specify exactly one of catalog_number, international_designator, or name"
            )
        param_name = param_names[0]
        param_value = params[param_name]
        url = (
            "https://celestrak.com/NORAD/elements/gp.php?"
            f"{param_name}={param_value}"
            "&FORMAT=XML"
        )
        return url
    
    def _segments_from_query(url):
        response = httpx.get(url)
        response.raise_for_status()
    
        if response.text == "No GP data found":
            raise ValueError(
                f"Query '{url}' did not return any results, try a different one"
            )
        tree = ET.parse(io.StringIO(response.text))
        root1 = tree.getroot()
    
        yield from omm.parse_xml(io.StringIO(response.text))
    
    def load_gp_from_celestrak(
        *, catalog_number=None, international_designator=None, name=None
    ):
        """Load general perturbations orbital data from Celestrak.
    
        Returns
        -------
        Satrec
            Orbital data from specified object.
    
        Notes
        -----
        This uses the OMM XML format from Celestrak as described in [1]_.
    
        References
        ----------
        .. [1] Kelso, T.S. "A New Way to Obtain GP Data (aka TLEs)"
           https://celestrak.com/NORAD/documentation/gp-data-formats.php
    
        """
        # Assemble query, raise an error if malformed
        url = _generate_url(catalog_number, international_designator, name)
    
        # Make API call, raise an error if data is malformed
        for segment in _segments_from_query(url):
            # Initialize and return Satrec object
            sat = Satrec()
            omm.initialize(sat, segment)
    
            yield sat
    
    def print_sat(sat, name):
        """Prints Satrec object in convenient form."""
        print(json.dumps(exporter.export_omm(sat, name), indent=2))
    
    #this function calculates the angle between two vectors
    def vec_angle(a,b):
        inner = np.inner(a,b)
        norms = LA.norm(a)*LA.norm(b)
        cos = inner/norms
        rad = np.arccos(np.clip(cos, -1.0, 1.0))
        deg = np.rad2deg(rad)
        return deg
    
    #write a function to convert GCRS to RIC
    def GCRS_to_RIC(rt, vt, ri, vi):
        #rt and vt is the center of the new RIC reference frame
        
        r_units = ri.unit
        v_units = vi.unit
        
        rt = rt.value
        vt = vt.value
        ri = ri.value
        vi = vi.value
            
        R_hat = rt/LA.norm(rt)
        I_hat = vt/LA.norm(vt)
        C_cross = np.cross(R_hat, I_hat)
        C_hat = C_cross/LA.norm(C_cross)
        
        r = ri - rt
        v = vi - vt
        
        rRIC = [np.dot(r, R_hat), np.dot(r, I_hat), np.dot(r, C_hat)]
        vRIC = [np.dot(v, R_hat), np.dot(v, I_hat), np.dot(v, C_hat)]
        
        RBack = [np.dot([1,0,0], R_hat), np.dot([1,0,0], I_hat), np.dot([1,0,0], C_hat)]
        IBack = [np.dot([0,1,0], R_hat), np.dot([0,1,0], I_hat), np.dot([0,1,0], C_hat)]
        CBack = [np.dot([0,0,1], R_hat), np.dot([0,0,1], I_hat), np.dot([0,0,1], C_hat)]
        
        return rRIC*r_units, vRIC*v_units, RBack, IBack, CBack
    
    def ellipse_est(Rp, Rm, Ip, Im):
        
        i_plus_last = Ip.shape[0]-1
        i_middle_plus = int(Ip.shape[0]/2)
        i_middle_minus = int(Im.shape[0]/2)
    
        semi1 = np.sqrt((Ip[i_plus_last] - Ip[0])**2+(Rp[i_plus_last] - Rp[0])**2)/2
        semi2 = np.sqrt((Ip[i_middle_plus] - Im[i_middle_minus])**2+(Rp[i_middle_plus] - Rm[i_middle_minus])**2)/2
        area = (semi1)*(semi2)*np.pi
        
        return semi1, semi2, area
    
    def RIC_cone_proj(r, v, angle_degrees, I_span, C):
        #returns the R coordinates of a half cone projection into the RIC Plane
        # r: position of the cone
        # v: vector the cone points at
        # angle_degrees: angle of the cone
        # i_span: the values of the x coordinate we are giving back ys for
        # C: plane of the projection
        
        Ra = r[0]
        Ia = r[1]
        Ca = r[2]
        
        I = I_span
        
        angle_rad = angle_degrees * np.pi/180
        
        v_unit = (v / LA.norm(v))
        
        vR = v_unit[0]
        vI = v_unit[1]
        vC = v_unit[2]
        
        
        cos2theta = np.cos(angle_rad)*np.cos(angle_rad)
        
        alpha = vI*(I-Ia)+vC*(C-Ca) - vR*Ra
        
        beta = (I-Ia)**2+(C-Ca)**2
        
        delta = alpha**2 - beta*cos2theta
        
        a = (vR*vR - cos2theta)
        
        b = (2*vR*alpha+2*Ra*cos2theta)
        
        c = delta - Ra*Ra*cos2theta
        
        # run block of code and catch warnings
        with warnings.catch_warnings():
            #ignoare sqrt of negative number warning
                            
            warnings.filterwarnings("ignore")
            
            R_plus = (-b+np.sqrt(b*b-4*a*c))/(2*a)
        
            R_minus = (-b-np.sqrt(b*b-4*a*c))/(2*a)
        
    
        #Filuter out the NaN
        i_plus = ~np.isnan(R_plus)
        i_minus = ~np.isnan(R_minus)
        I_plus = I[i_plus]
        I_minus = I[i_minus]
        R_plus = R_plus[i_plus]
        R_minus = R_minus[i_minus]
        
        #filter out the bottom half of the cone
        Plus_dot = vR * (R_plus - Ra) + vI * (I_plus - Ia) + vC * (C - Ca)
        Minus_dot = vR * (R_minus - Ra) + vI * (I_minus - Ia) + vC * (C - Ca)
    
        i_plus_pos = Plus_dot > 0
        i_minus_pos = Minus_dot > 0
    
        I_plus = I_plus[i_plus_pos]
        I_minus = I_minus[i_minus_pos]
        R_plus = R_plus[i_plus_pos]
        R_minus = R_minus[i_minus_pos]
            
        return I_plus, I_minus, R_plus, R_minus
    ###############################################################################
    ######################### Command Functions Go Here ###########################
    
    ################################# Update Now Global
    def update_now_global(*args):
        
        global global_now
        
        global_now = Time.now()
        
        c_launch_epoch.set(str(global_now.datetime))
        
        update_target_global()
        
        return
    
    
    ########################################### Update Target Orbit Global ############
    def update_target_global(*args):
        
        global target, global_now
        
      
        target = list(load_gp_from_celestrak(name=c_target.get()))[0]
    
        #call the update launch time
        update_launch_time_global()
        
    def update_launch_time_global(*args):
        
        global global_target_orb, target
        
        L0 = global_now+c_l_t_minus.get()*u.h
        
        c_launch_time.set(str(L0.datetime))
        
        error, r, v = target.sgp4(L0.jd1, L0.jd2)
    
        #teme = TEME(r, obstime=L0)
    
        #gcrs = GCRS(obstime=L0)
    
        global_target_orb = Orbit.from_vectors(Earth, r*u.km, v*u.km/u.s, epoch=L0)
        
        update_asat_global()
        
        endgame_analysis()
    
        
    ########################################### Update Global ASAT ############  
    def update_asat_global(*args):
        
        global global_target_orb, global_target_impact_orb, global_asat_orb
        global global_asat_impact_orb, global_t_end, global_asat_endgame_orb
        
        
        tof = c_TOF.get()*u.h
        
        L0 = global_now+c_l_t_minus.get()*u.h
    
        
        #extract lat/lon for the GUI
        asat_lon = c_site_lon.get()*u.deg
        
        asat_lat = c_site_lat.get()*u.deg
    
        asat_alt = 0*u.m    
        
        #get the inertial position of the launch site
        asat_site = coord.EarthLocation(asat_lon, asat_lat, asat_alt)
        
        #convert to GCRS
        gcrs = asat_site.get_gcrs(obstime=global_now)
        
        #extract position and velocity of the launch site
        r_asat_L = gcrs.cartesian.xyz<<u.km
        v_asat_L = gcrs.velocity.d_xyz
        
        #output earth assist
        c_earth_sp.set(str(np.around(LA.norm(v_asat_L), 2)))        
        
        #propogate target orbit to its position at TOF
        global_target_impact_orb = global_target_orb.propagate(tof)
    
        #get Lambert Solution data
        k = Earth.k.to(u.km ** 3 / u.s ** 2)
        r0 = r_asat_L
        r1 = global_target_impact_orb.r
    
        #take the shot
        v1, v2 = vallado(k.value, r0, r1, tof*60*60, short=True, numiter=100, rtol=1e-8)
        
        #output launch delta-v
        total_deltav = (v1- v_asat_L.value)*u.km/u.s
        c_launch_sp.set(str(np.around(LA.norm(total_deltav), 2)))        
    
        #create ASAT orbit
        global_asat_orb = Orbit.from_vectors(Earth, r0, v1*u.km/u.s, epoch=L0)
        c_launch_inc.set(str(np.around(global_asat_orb.inc<<u.deg, 2)))
        c_apogee.set(str(np.around(global_asat_orb.r_a<<u.km, 2)))
                         
        
        #propogate target orbit to its position at TOF
        global_asat_impact_orb = global_asat_orb.propagate(tof)
        
        #calculate closing data and endgame time
        r_end = c_KKV_acq.get()*u.km
        closing_v = LA.norm(global_asat_impact_orb.v-global_target_impact_orb.v)
        global_t_end = tof - r_end / closing_v
        dt_end = tof-global_t_end
        global_asat_endgame_orb = global_asat_orb.propagate(global_t_end)
    
        #get the ephemerides for the solid line plot
        epochs = time_range(L0, end=L0+tof)
        asat_e = Ephem.from_orbit(global_asat_orb, epochs, plane=Planes.EARTH_EQUATOR)
        target_e = Ephem.from_orbit(global_target_orb, epochs, plane=Planes.EARTH_EQUATOR)
        
        
        #create the inertial window
        fig_inertial = plt.Figure(figsize=(size_x,size_y), dpi=100)
        chart_type_inertial = FigureCanvasTkAgg(fig_inertial, root)
        chart_type_inertial.get_tk_widget().grid(column = 2, row = 0)   
        ax_inertial = fig_inertial.add_subplot(111)
    
        #plot the inertial trajectories
        frame = StaticOrbitPlotter(ax_inertial, plane=Planes.EARTH_EQUATOR)
        frame.plot(global_target_orb, label="Target at L = 0")
        frame.plot(global_asat_orb, label="ASAT at L = 0", color="red")
        frame.plot_ephem(asat_e, label="ASAT Trajectory", color="red")
        frame.plot_ephem(target_e, label="Target Trajectory", color="blue")
        ax_inertial.get_legend().remove()
        ax_inertial.set_title('DA ASAT Intertial Trajectory')
        ax_inertial.yaxis.set_label_position("right")
    
        
        update_endgame_windows()
        
    def update_endgame_windows(*args):
        
        global global_target_orb, global_target_impact_orb, global_asat_orb
        global global_asat_impact_orb, global_t_end, global_asat_endgame_orb
           
        #load variables that will be needed for calulcations
        tof = c_TOF.get()*u.h       
        r_end = c_KKV_acq.get()*u.km
        FOV = c_KKV_FOV.get()*u.deg
        L0 = global_now+c_l_t_minus.get()*u.h
        CTS_end = c_KKV_CTS.get()*u.deg
        max_divert = c_KKV_max_divert.get()* u.m/u.s<<u.km/u.s
        
        #Propogate the target orbit to the time of the threat reaction
        t_tr = c_t_tr.get()*u.h
        target_orb_tr = global_target_orb.propagate(t_tr)
        r_t_tr = target_orb_tr.r
        v_t_tr = target_orb_tr.v
          
        #get the threat reaction magnitude
        dv_tr = [c_R_tr_v.get(), c_I_tr_v.get(), c_C_tr_v.get()]*u.m/u.s << u.km/u.s
        #update the vector magnitude
        c_tr_mag.set(str(np.around(LA.norm(dv_tr)<<u.m/u.s, 2)))
          
        # get the target RIC at the time of the threat reactoin
        rRIC_tr, vRIC_tr, i, j, k = GCRS_to_RIC(r_t_tr, v_t_tr, r_t_tr, dv_tr)
        
        #convert tr_dv from RIC to FCRS
        tr_dv_inertial = [np.dot(i, dv_tr).value, np.dot(j, dv_tr).value, np.dot(k, dv_tr).value]*dv_tr.unit
        
        #create and apply the maneuver object to the satellite
        dv_impulse = 0*u.s
        man = Maneuver((dv_impulse, tr_dv_inertial))
        target_orb_new = target_orb_tr.apply_maneuver(man)
        
        #calculate how long the endgame is
        dt_end = tof-global_t_end
        c_endgame_dt.set(str(np.around(dt_end<<u.min, 1)))
        asat_orb_endgame = global_asat_orb.propagate(global_t_end)
        target_orb_endgame_org = global_target_orb.propagate(global_t_end)
        target_orb_endgame_new = target_orb_new.propagate(global_t_end-t_tr)
        
        # get the vectors for ASAT RIC --> GCRS (i, j, k)
        rRIC_asat, vRIC_asat, i, j, k = GCRS_to_RIC(target_orb_endgame_org.r, target_orb_endgame_org.v, asat_orb_endgame.r, asat_orb_endgame.v)
    
        # get the vectors for new position RIC --> GCRS (i, j, k)
        rRIC_new, vRIC_new, i, j, k = GCRS_to_RIC(target_orb_endgame_org.r, target_orb_endgame_org.v, target_orb_endgame_new.r, target_orb_endgame_new.v)
    
        #update output values
        c_impact_sp.set(str(np.around(LA.norm(vRIC_asat), 1)))
        c_KKV_r.set(str(np.around(rRIC_asat, 1)))
        c_KKV_v.set(str(np.around(vRIC_asat, 1)))
        c_blue_r.set(str(np.around(rRIC_new, 1)))
        c_blue_v.set(str(np.around(vRIC_new<<u.m/u.s, 1)))
        
        #calculate divert required
        v1 = vRIC_asat
        v2 = (rRIC_new+vRIC_new*(dt_end<<u.s)) - rRIC_asat
        theta = vec_angle(v1, v2)
        divert_dv = np.tan(theta<<u.rad)*LA.norm(vRIC_asat)<<u.m/u.s
        c_divert_required.set(str(np.around(divert_dv, 2)))
    
        ###########################################################################
        ############# RI Plane
        #create the RI Window
        fig_RI = plt.Figure(figsize=(size_x,size_y), dpi=100)
        chart_type_RI = FigureCanvasTkAgg(fig_RI, root)
        chart_type_RI.get_tk_widget().grid(column = 3, row = 0, padx = graphpadx)   
        ax_RI = fig_RI.add_subplot(111)
        
        lim = r_end.value
        
        ax_RI.plot([0, 0], [lim, -lim], color='lightgray', linestyle='--', zorder = 3)
        ax_RI.plot([lim, -lim], [0, 0], color='lightgray', linestyle='--', zorder = 3)
        ax_RI.set_xlim(lim, -lim)
        ax_RI.set_ylim(-lim, lim)
        ax_RI.yaxis.set_label_position("right")
        ax_RI.set_xlabel("In-track Position (km - Velocity Points Left)")
        ax_RI.set_ylabel("Radial Position (km - Earth is Down)")
        title = "Endgame Geometry RI-Plane\n C = Blue HVA Position ("+str(np.around(rRIC_new[2], 1))+")"
        ax_RI.set_title(title)
    
        
        ##############################    FOV
        #plot the field of view cone intersection
    
        #Get the Intrack point for evaluating the FOV
        I_span_fov = np.linspace(-5*r_end, 5*r_end, num=10000)
    
        #Project the cone into the C=0 plane and calculate the dimensions
        I_plus_fov, I_minus_fov, R_plus_fov, R_minus_fov = RIC_cone_proj(rRIC_asat, vRIC_asat, FOV.value/2, I_span_fov, rRIC_new[2])
        FOV_semi1, FOV_semi2, FOV_area = ellipse_est(R_plus_fov, R_minus_fov, I_plus_fov, I_minus_fov)
        FOV_str = str(np.around(FOV_semi1*2, 2))+" X " + str(np.around(FOV_semi2*2, 2))
        c_FOV_size.set(FOV_str)
        
        
        #Plot the KE WEZ
        #code combines and flips plots so that the whole circle is filled
        ax_RI.plot(np.hstack([I_plus_fov, np.flip(I_minus_fov), I_plus_fov[0]]), np.hstack([R_plus_fov, np.flip(R_minus_fov),R_plus_fov[0]]), color='black', zorder = 3)
        ax_RI.fill(np.hstack([I_plus_fov, np.flip(I_minus_fov)]), np.hstack([R_plus_fov, np.flip(R_minus_fov)]), color='whitesmoke', zorder = 2, alpha = .5)
        
        ################################   SUN
        #plot the sun cone intersection
        #use astropy to get the sun position at now+t_end
        sun = coord.get_sun(L0+global_t_end)
        r_sun = sun.cartesian.xyz<<u.km
        rRIC_sun, vRIC_sun, i, j, k = GCRS_to_RIC(target_orb_endgame_org.r, target_orb_endgame_org.v, r_sun, [0,0,0]*u.km/u.s)
        sun_to_asat = rRIC_asat - rRIC_sun
        cts_at_endgame = vec_angle(rRIC_sun - rRIC_new, rRIC_asat - rRIC_new)
        
        c_CTS_endgame.set(str(np.around(cts_at_endgame, 1)))
        
        #Get the Intrack point for evaluating the sun
        I_span = np.linspace(-5*r_end, 5*r_end, num=10000)
    
        #Project the cone into the C=0 plane
        I_plus_sun, I_minus_sun, R_plus_sun, R_minus_sun = RIC_cone_proj(rRIC_asat, sun_to_asat, CTS_end.value, I_span, rRIC_new[2])
        ax_RI.fill(np.hstack([I_plus_sun, np.flip(I_minus_sun)]), np.hstack([R_plus_sun, np.flip(R_minus_sun)]), color='yellow', zorder = 1)
    
        ################################   KE WEZ
        #calculate WEZ
    
        #calculate half-angle of the KE WEZ cone
        deg = np.arctan(max_divert/LA.norm(vRIC_asat)).value*180/np.pi
    
        #Get the Intrack point for evaluating the KE WEZ
        I_span = np.linspace(-r_end*5, r_end*5, num=1000)
    
        #Project the cone into the C=0 plane
        I_plus, I_minus, R_plus, R_minus = RIC_cone_proj(rRIC_asat, vRIC_asat, deg, I_span, 0)
        KE_semi1, KE_semi2, KE_area = ellipse_est(R_plus, R_minus, I_plus, I_minus)
        KE_str = str(np.around(FOV_semi1*2, 2))+" X " + str(np.around(FOV_semi2*2, 2))
        c_KE_WEZ_size.set(KE_str)
    
        #Calculate look angle to HVA
        v1 = -rRIC_asat
        v2 = rRIC_new - rRIC_asat
        theta = np.around(vec_angle(v1, v2), 2)
        c_look.set(theta)
        
        
        #Plot the KE WEZ
        ax_RI.plot(np.hstack([I_plus, np.flip(I_minus), I_plus[0]]), np.hstack([R_plus, np.flip(R_minus), R_plus[0]]), color='red', zorder = 3)
        
        ##########################    Plot location points
        #plot all the points
        org_target_label = ax_RI.scatter(0, 0, color='dimgray', marker = "X", zorder = 4)
        new_target_label = ax_RI.scatter(rRIC_new[1], rRIC_new[0], color='blue', zorder = 4)
        asat_label = ax_RI.scatter(rRIC_asat[1], rRIC_asat[0], color='red', zorder = 4)
        fov = ax_RI.scatter(rRIC_sun[1], rRIC_sun[0], color='whitesmoke', marker = "<", s = 150, zorder = 4, edgecolors = 'black')
        sun = ax_RI.scatter(rRIC_sun[1], rRIC_sun[0], color='yellow', marker = "<", s = 150, zorder = 3, edgecolors = 'black')
        wez_legend_line = ax_RI.scatter(rRIC_sun[1], rRIC_sun[0], color='red', marker = "_", s = 150, zorder = 3)
        
        ######################       Plot Legend Box
        legend1 = ax_RI.legend([org_target_label, new_target_label, asat_label, fov, sun, wez_legend_line], ['ASAT Aim Point', 'Target Location', 'KKV'], loc='upper right', framealpha = 1)
        legend2 = ax_RI.legend([fov, sun, wez_legend_line], ['KKV FOV', 'KKV EO-WEZ', 'KKV KE-WEZ'], loc='lower right', framealpha = 1)
        ax_RI.add_artist(legend1)
        ax_RI.add_artist(legend2)
        plt.tight_layout()
        
        ###########################################################################
        ############# CI Plane
        #Most of these calculations are the same, except the frame is now CIR rather than RIC. Hence np.flip()
        #create the RI Window
        fig_CI = plt.Figure(figsize=(size_x,size_y), dpi=100)
        chart_type_CI = FigureCanvasTkAgg(fig_CI, root)
        chart_type_CI.get_tk_widget().grid(column = 3, row = 1, padx = graphpadx)   
        ax_CI = fig_CI.add_subplot(111)
        lim = r_end.value
        ax_CI.plot([0, 0], [lim, -lim], color='lightgray', linestyle='--', zorder = 3)
        ax_CI.plot([lim, -lim], [0, 0], color='lightgray', linestyle='--', zorder = 3)
        ax_CI.set_xlim(lim, -lim)
        ax_CI.set_ylim(-lim, lim)
        ax_CI.yaxis.set_label_position("right")
        ax_CI.set_xlabel("In-track Position (km - Velocity Points Left)")
        ax_CI.set_ylabel("Cross-track Position (km - North is Up)")
        title = "Endgame Geometry CI-Plane\n R = Blue HVA Position ("+str(np.around(rRIC_new[0], 1))+")"
        ax_CI.set_title(title)
    
        ##############################    FOV
        #plot the field of view cone intersection
    
        #Project the cone into the R=0 plane and calculate the dimensions
        I_plus_fov, I_minus_fov, C_plus_fov, C_minus_fov = RIC_cone_proj(np.flip(rRIC_asat), np.flip(vRIC_asat), FOV.value/2, I_span_fov, rRIC_new[0])
        #Plot the KE WEZ
        #code combines and flips plots so that the whole circle is filled
        ax_CI.plot(np.hstack([I_plus_fov, np.flip(I_minus_fov), I_plus_fov[0]]), np.hstack([C_plus_fov, np.flip(C_minus_fov),C_plus_fov[0]]), color='black', zorder = 3)
        ax_CI.fill(np.hstack([I_plus_fov, np.flip(I_minus_fov)]), np.hstack([C_plus_fov, np.flip(C_minus_fov)]), color='whitesmoke', zorder = 2, alpha = .5)
    
    
        #Project the sun cone into the R=0 plane
        I_plus_sun, I_minus_sun, C_plus_sun, C_minus_sun = RIC_cone_proj(np.flip(rRIC_asat), np.flip(sun_to_asat), CTS_end.value, I_span, rRIC_new[0])
        ax_CI.fill(np.hstack([I_plus_sun, np.flip(I_minus_sun)]), np.hstack([C_plus_sun, np.flip(C_minus_sun)]), color='yellow', zorder = 1)
    
        #Calculate the KE WEZ in the CI plane
        I_plus, I_minus, C_plus, C_minus = RIC_cone_proj(np.flip(rRIC_asat), np.flip(vRIC_asat), deg, I_span, rRIC_new[0])
        ax_CI.plot(np.hstack([I_plus, np.flip(I_minus), I_plus[0]]), np.hstack([C_plus, np.flip(C_minus), C_plus[0]]), color='red', zorder = 3)
    
        ##########################    Plot location points
        #plot all the points for the CI
        org_target_label = ax_CI.scatter(0, 0, color='dimgray', marker = "X", zorder = 4)
        new_target_label = ax_CI.scatter(rRIC_new[1], rRIC_new[2], color='blue', zorder = 4)
        asat_label = ax_CI.scatter(rRIC_asat[1], rRIC_asat[2], color='red', zorder = 4)
        fov = ax_CI.scatter(rRIC_sun[1], rRIC_sun[2], color='whitesmoke', marker = "<", s = 150, zorder = 4, edgecolors = 'black')
        sun = ax_CI.scatter(rRIC_sun[1], rRIC_sun[2], color='yellow', marker = "<", s = 150, zorder = 3, edgecolors = 'black')
        wez_legend_line = ax_CI.scatter(rRIC_sun[1], rRIC_sun[2], color='red', marker = "_", s = 150, zorder = 3)
        
        ######################       Plot Legend Box
        legend1 = ax_CI.legend([org_target_label, new_target_label, asat_label, fov, sun, wez_legend_line], ['ASAT Aim Point', 'Target Location', 'KKV'], loc='upper right', framealpha = 1)
        legend2 = ax_CI.legend([fov, sun, wez_legend_line], ['KKV FOV', 'KKV EO-WEZ', 'KKV KE-WEZ'], loc='lower right', framealpha = 1)
        ax_CI.add_artist(legend1)
        ax_CI.add_artist(legend2)
    
    def endgame_analysis():
        
        global global_now, global_target_orb
        
        t_steps = 44
        tof_min = 2*u.h
        tof_max = 24*u.h
        tof_range = np.linspace(start = tof_min, stop=tof_max, num=t_steps, endpoint=True)
        
        L0 = global_now+c_l_t_minus.get()*u.h
    
        r_end = c_KKV_acq.get()*u.km
        
        CTS_end = c_KKV_CTS.get()*u.deg
        
        #extract lat/lon for the GUI
        asat_lon = c_site_lon.get()*u.deg
        
        asat_lat = c_site_lat.get()*u.deg
    
        asat_alt = 0*u.m    
        
        #get the inertial position of the launch site
        asat_site = coord.EarthLocation(asat_lon, asat_lat, asat_alt)
        
        #convert to GCRS
        gcrs = asat_site.get_gcrs(obstime=L0)
        
        #extract position and velocity of the launch site
        r_asat_L = gcrs.cartesian.xyz<<u.km
        v_asat_L = gcrs.velocity.d_xyz
        
        
        #load loop constants
        k = Earth.k.to(u.km ** 3 / u.s ** 2)
        r0 = r_asat_L
        
        #create data vectors
        deltaV_t = np.array([])
        CTS_t = np.array([])
        t_closing_t = np.array([])
        R_closing = np.array([])
        I_closing = np.array([])
        C_closing = np.array([])
        
        for t in tof_range:
            
            #propagate target orbit to time t
            global_target_orb_impact = global_target_orb.propagate(t)
            r_t = global_target_orb_impact.r
            v_t = global_target_orb_impact.v
        
            #solve Lambert Equation
            v_asat_0, v_asat_1 = vallado(k.value, r0, r_t, t*60*60, short=True, numiter=100, rtol=1e-8)
        
            #convert velocities to quanity
            v_asat_0=v_asat_0*u.km/u.s
            v_asat_1=v_asat_1*u.km/u.s
            
            #using closing velocty to estimate t_end (when is terminal acquisition)
            closing_v = LA.norm(v_asat_1-v_t)
            t_end = t - r_end / closing_v
            t_closing_t = np.hstack((t_closing_t, r_end / closing_v))
            
            #create ASAT Orbit
            asat_orb = Orbit.from_vectors(Earth, r0, v_asat_0, epoch=L0)
            deltaV_t = np.hstack((deltaV_t, LA.norm(v_asat_0-v_asat_L.value*u.km/u.s)))
                
            #calculate camerata-target-sun
            #propagate orbits to endgame
            asat_end = asat_orb.propagate(t_end)
            target_end = global_target_orb.propagate(t_end)
        
            #vectors for asat interceptor (i) and target (t)
            r_i = asat_end.r
            v_i = asat_end.v
            r_t = target_end.r
            v_t = target_end.v
        
            #use astropy to get the sun position at now+t_end
            sun = coord.get_sun(L0+t_end)
            r_sun = sun.cartesian.xyz<<u.km
        
            #calulate the target-to-sun position vector and 
            #target-to-interceptor position vector
            t2sun = r_sun - r_t
            t2i = r_i - r_t
            cts_end = vec_angle(t2sun, t2i)
        
            CTS_t = np.hstack((CTS_t, cts_end))
            
            #determine position at intercept
            rRIC, vRIC, iback, jback, kback = GCRS_to_RIC(r_t, v_t, r_i, v_i)
            R_closing = np.hstack((R_closing, rRIC[0]))
            I_closing = np.hstack((I_closing, rRIC[1]))
            C_closing = np.hstack((C_closing, rRIC[2]))
    
        #determine when there is good lightning and when there is bad lightning
        CTS_good_light = np.where(CTS_t<=CTS_end)
        CTS_bad_light = np.where(CTS_t>CTS_end)
    
        #create the RI Window
        fig_endgame = plt.Figure(figsize=(size_x,size_y), dpi=100)
        chart_type_endgame = FigureCanvasTkAgg(fig_endgame, root)
        chart_type_endgame.get_tk_widget().grid(column = 2, row = 1)   
        ax_endgame = fig_endgame.add_subplot(111)
        
    
        good_lighting = ax_endgame.scatter(tof_range[CTS_good_light], t_closing_t[CTS_good_light]/60, color='goldenrod')
        bad_lighting = ax_endgame.scatter(tof_range[CTS_bad_light], t_closing_t[CTS_bad_light]/60, color='dimgray')
        ax_endgame.set_xlabel("ASAT Time of Flight (hours)")
        ax_endgame.set_ylabel("End Game Duration (Minutes)")
        title = "Terminal Acquisition Timelines"
        ax_endgame.set_title(title)
        legend2 = ax_endgame.legend([good_lighting, bad_lighting], ['Good Lighting', 'Bad Lighting'], loc='best', framealpha = .5)
        ax_endgame.yaxis.set_label_position("right")
    
    
        
    def open_popup():
    
        endgame_analysis()
    
        top= tk.Toplevel(root)
        top.geometry("600x600")
        top.title("ASAT Shot Windows")
       
        #extract lat/lon for the GUI
        asat_lon = c_site_lon.get()*u.deg
        asat_lat = c_site_lat.get()*u.deg
        asat_alt = 0*u.m    
        
        #convert asat launch site lat/lon to GCRS
        asat_site = coord.EarthLocation(asat_lon, asat_lat, asat_alt)
        
        #use astropy library to transform from lat/lon to gcrs
        gcrs = asat_site.get_gcrs(obstime=global_now)
        
        #extract cartesian position
        r_asat_L = gcrs.cartesian.xyz<<u.km
        
        #extract velocity at site
        v_asat_L = gcrs.velocity.d_xyz
        
        error, r, v = target.sgp4(global_now.jd1, global_now.jd2)
        teme = TEME(r, obstime=global_now)
        gcrs = GCRS(obstime=global_now)
        target_orb = Orbit.from_vectors(Earth, r*u.km, v*u.km/u.s, epoch=global_now)
        
        #load loop constants
        k = Earth.k.to(u.km ** 3 / u.s ** 2)
        
        
        #create the inertial window
        fig_window = plt.Figure(figsize=(6,6), dpi=100)
        chart_type_window = FigureCanvasTkAgg(fig_window, top)
        chart_type_window.get_tk_widget().grid(column = 0, row = 0)   
        ax_window = fig_window.add_subplot(111)
        
        #how long to run the window
        win_steps = 25
        win_min = 0
        win_max = 24
        win_range = np.linspace(start = win_min, stop=win_max, num=win_steps, endpoint=True)
        
        #Time Window to investigate. Simulation will find the optimal launch time from now un
        t_steps = 40
        tof_min = 2*u.h
        tof_max = 15*u.h
        tof_range = np.linspace(start = tof_min, stop=tof_max, num=t_steps, endpoint=True)
        
        CTS_end = c_KKV_CTS.get()*u.deg
        r_end = c_KKV_acq.get() * u.km
           
        for w in win_range:
            
        
            #create data vectors for now shot
            deltaV_t = np.array([])
            CTS_t = np.array([])
            R_closing = np.array([])
            I_closing = np.array([])
            C_closing = np.array([])
            t_closing_t =([])
            win_now = global_now+w*u.h
              
            #convert asat launch site lat/lon to GCRS
            asat_site = coord.EarthLocation(asat_lon, asat_lat, asat_alt)
        
            #use astropy library to transform from lat/lon to gcrs
            gcrs = asat_site.get_gcrs(obstime=win_now)
        
            #extract cartesian position
            r_asat_L = gcrs.cartesian.xyz<<u.km
            r0 = r_asat_L
        
            #extract velocity at site
            v_asat_L = gcrs.velocity.d_xyz
            
            #get r and v of target at now
            error, r, v = target.sgp4(win_now.jd1, win_now.jd2)
            teme = TEME(r, obstime=win_now)
            gcrs = GCRS(obstime=win_now)
            target_orb = Orbit.from_vectors(Earth, r*u.km, v*u.km/u.s, epoch=win_now)
        
            for t in tof_range:
        
                #propagate target orbit to time t
                target_orb_impact = target_orb.propagate(t)
                r_t = target_orb_impact.r
                v_t = target_orb_impact.v
        
                #solve Lambert Equation
                v_asat_0, v_asat_1 = vallado(k.value, r0, r_t, t*60*60, short=True, numiter=100, rtol=1e-8)
        
                #convert velocities to quanity
                v_asat_0=v_asat_0*u.km/u.s
                v_asat_1=v_asat_1*u.km/u.s
        
                #using closing velocty to estimate t_end (when is terminal acquisition)
                closing_v = LA.norm(v_asat_1-v_t)
                t_end = t - r_end / closing_v
                t_closing_t = np.hstack((t_closing_t, r_end / closing_v))
        
                #create ASAT Orbit
                asat_orb = Orbit.from_vectors(Earth, r0, v_asat_0, epoch=win_now)
                deltaV_t = np.hstack((deltaV_t, LA.norm(v_asat_0-v_asat_L.value*u.km/u.s)))
        
                #calculate camerata-target-sun
                #propagate orbits to endgame
                asat_end = asat_orb.propagate(t_end)
                target_end = target_orb.propagate(t_end)
        
                #vectors for asat interceptor (i) and target (t)
                r_i = asat_end.r
                v_i = asat_end.v
                r_t = target_end.r
                v_t = target_end.v
        
                #use astropy to get the sun position at now+t_end
                sun = coord.get_sun(win_now+t_end)
                r_sun = sun.cartesian.xyz<<u.km
        
                #calulate the target-to-sun position vector and 
                #target-to-interceptor position vector
                t2sun = r_sun - r_t
                t2i = r_i - r_t
                cts_end = vec_angle(t2sun, t2i)
        
                CTS_t = np.hstack((CTS_t, cts_end))
        
                #determine position at intercept
                
                rRIC, vRIC, RBack, IBack, CBack = GCRS_to_RIC(r_t, v_t, r_i, v_i)
                R_closing = np.hstack((R_closing, rRIC[0]))
                I_closing = np.hstack((I_closing, rRIC[1]))
                C_closing = np.hstack((C_closing, rRIC[2]))
        
            #get the windows with good lighting
            CTS_good_light = np.where(CTS_t<=CTS_end)
            w_graph = np.ones(tof_range[CTS_good_light].size)*w
        
            #calculate max endgame
           
            try:
                end_max = np.max(t_closing_t[CTS_good_light])    
                end_max_index = np.where(t_closing_t[CTS_good_light]==end_max)
                good_light = ax_window.scatter(w_graph, tof_range[CTS_good_light], color='goldenrod')
                min_endgame = ax_window.scatter(w, tof_range[CTS_good_light][end_max_index], color='red')
            except:
                Reagan = 13
    
        ax_window.set_xlabel("L0 Time of Launch (Now+HH hrs)")
        ax_window.set_ylabel("ASAT Time of Flight (hours)")
        title = "CTS Launch Windows\nRed L0-TOF Pair Maximizes Endgame Duration"
        ax_window.set_title(title)
        legend2 = ax_window.legend([good_light, min_endgame], ['Good Lighting', 'Max Endgame'], loc='best', framealpha = .5)
        ax_window.set_xlim(0, 24)
    
###############################################################################
########################## Start Drawing the GUI ##############################

    root = tk.Tk()
    root.title('GEO-class DA ASAT Tool')
    root.geometry('1500x750')
    
    ################## Create the Engagement Frame ################################
    #the prefix c_ on a variable represents a contorl variable
    padx_ef = 5
    pady_ef = 5
    
    engagement_frame = ttk.Frame(root)
    engagement_frame.columnconfigure(0, weight=1)
    engagement_frame.columnconfigure(1, weight=1)
    
    # (Row 0) target input
    ttk.Label(engagement_frame, text='Target').grid(column=0, row=0, sticky=tk.E)
    c_target = tk.StringVar()
    c_target.set("AEHF-6 (USA 298)")
    target_drop = tk.OptionMenu(engagement_frame, c_target, 
                                "AEHF-1 (USA 214)",
                                "AEHF-2 (USA 235)",
                                "AEHF-3 (USA 246)",
                                "AEHF-4 (USA 288)", 
                                "AEHF-5 (USA 292)",
                                "AEHF-6 (USA 298)",
                                "SBIRS GEO-1 (USA 230)",
                                "SBIRS GEO-2 (USA 241",
                                "SBIRS GEO-3 (USA 282)",
                                "SBIRS GEO-4 (USA 273)",
                                "SBIRS GEO-5 (USA 315)",
                                "WGS F1 (USA 195)", 
                                "WGS F2 (USA 204)",
                                "WGS F3 (USA 211)",
                                "WGS F4 (USA 233)",
                                "WGS F5 (USA 243)",
                                "WGS F6 (USA 244)",
                                "WGS F7 (USA 263)",
                                "WGS F8 (USA 272)",
                                "WGS F9 (USA 275)",
                                "WGS 10 (USA 291)",
                                command = update_target_global)
    target_drop.grid(column = 1, row = 0, pady = pady_ef)
    
    # (Row 1) Create Latitutde inputs
    ttk.Label(engagement_frame, text='ASAT Launch Site Latitude (degrees):').grid(column=0, row=1, sticky=tk.E)
    c_site_lat = tk.DoubleVar()
    c_site_lat.set(28)
    scroll_site_lat = tk.Scale(engagement_frame, from_=-90, to = 90, orient ='horizontal', variable = c_site_lat, command=update_asat_global)
    scroll_site_lat.grid(column=1, row=1, pady = pady_ef)
    
    # (Row 2) Create Longitude input
    ttk.Label(engagement_frame, text='ASAT Launch Site Longitude (degrees):').grid(column=0, row=2, sticky=tk.E)
    c_site_lon = tk.DoubleVar()
    c_site_lon.set(102)
    scroll_site_lon = tk.Scale(engagement_frame, from_=0, to = 359, orient ='horizontal', variable = c_site_lon, command = update_asat_global)
    scroll_site_lon.grid(column=1, row=2, pady = pady_ef)
    
    # (Row 3) Create ASAT Launch Time input
    ttk.Label(engagement_frame, text='ASAT Launch Time (Epoch + x hours):').grid(column=0, row=3, sticky=tk.E)
    c_l_t_minus = tk.DoubleVar()
    c_l_t_minus.set(0)
    scroll_l_t_minus = tk.Scale(engagement_frame, from_=0, to = 24, orient ='horizontal', variable = c_l_t_minus, resolution = 0.25, digits=4, command = update_launch_time_global)
    scroll_l_t_minus.grid(column = 1, row = 3, pady = pady_ef)
    
    # (Row 4) Create Set Epoch Button
    epoch_btn = ttk.Button(engagement_frame, text='Set Epoch', command = update_now_global)
    epoch_btn.grid(column=0, row = 4, pady = pady_ef)
    
    # (Row 4) Create Get Windows Button
    epoch_btn = ttk.Button(engagement_frame, text='Get Shot Windows', command = open_popup)
    epoch_btn.grid(column=1, row = 4, pady = pady_ef)
    
    # (Row 5) Create Time of Flight Slider
    ttk.Label(engagement_frame, text='ASAT Time of Flight in (hours):').grid(column=0, row=5, sticky=tk.E)
    c_TOF = tk.DoubleVar()
    c_TOF.set(5.15)
    scroll_TOF = tk.Scale(engagement_frame, from_=2, to = 20, orient ='horizontal', variable = c_TOF, resolution = 0.25, digits=4, command = update_asat_global)
    scroll_TOF.grid(column=1, row=5, pady = pady_ef)
    
    # (Row 6) KKV Acquisition Range Slider
    ttk.Label(engagement_frame, text='KKV Acquisition Range (kilometers):').grid(column=0, row=6, sticky=tk.E)
    c_KKV_acq=tk.DoubleVar()
    c_KKV_acq.set(500)
    scroll_KKV_acq = tk.Scale(engagement_frame, from_=200, to = 1000, orient ='horizontal', variable = c_KKV_acq, command = update_asat_global)
    scroll_KKV_acq.grid(column=1, row=6, pady = pady_ef)
    
    # (Row 7) KKV Max Divert
    ttk.Label(engagement_frame, text='KKV Max Divert (m/s):').grid(column=0, row=7, sticky=tk.E)
    c_KKV_max_divert = tk.DoubleVar()
    c_KKV_max_divert.set(400)
    scroll_KKV_max_divert = tk.Scale(engagement_frame, from_=100, to = 500, orient ='horizontal', variable = c_KKV_max_divert, command = update_endgame_windows)
    scroll_KKV_max_divert.grid(column=1, row=7, pady = pady_ef)
    
    # (Row 8) KKV Field of View
    ttk.Label(engagement_frame, text='KKV field of view (degrees):').grid(column=0, row=8, sticky=tk.E)
    c_KKV_FOV = tk.DoubleVar()
    c_KKV_FOV.set(10)
    scroll_KKV_FOV = tk.Scale(engagement_frame, from_=1, to = 10, orient ='horizontal', variable = c_KKV_FOV, command = update_endgame_windows)
    scroll_KKV_FOV.grid(column=1, row=8, pady = pady_ef)
    
    # (Row 9) KKV CTS Constraint
    ttk.Label(engagement_frame, text='KKV CTS constraint (degrees):').grid(column=0, row=9, sticky=tk.E)
    c_KKV_CTS = tk.DoubleVar()
    c_KKV_CTS.set(40)
    scroll_KKV_CTS = tk.Scale(engagement_frame, from_=1, to = 89, orient ='horizontal', variable = c_KKV_CTS, command = update_endgame_windows)
    scroll_KKV_CTS.grid(column=1, row=9, pady = pady_ef)
    
    # (Row 10) HVA Threat Reaction Time 
    # need to dynamically update this box whenever the TOF is changed
    ttk.Label(engagement_frame, text='HVA Threat Reaction Time (L + x hours):').grid(column=0, row=10, sticky=tk.E)
    c_t_tr=tk.DoubleVar()
    c_t_tr.set(1)
    scroll_t_tr = tk.Scale(engagement_frame, from_=1, to = 24, orient ='horizontal', variable = c_t_tr, resolution = 0.25, digits=4, command = update_endgame_windows)
    scroll_t_tr.grid(column=1, row=10, pady = pady_ef)
    
    # (Row 11) Radial Maneuver
    ttk.Label(engagement_frame, text='Radial threat reaction (m/s):').grid(column=0, row=11, sticky=tk.E)
    c_R_tr_v = tk.DoubleVar()
    c_R_tr_v.set(0)
    scroll_R_tr = tk.Scale(engagement_frame, from_=-20, to = 20, orient ='horizontal', variable = c_R_tr_v, resolution = .25, digits=4, command = update_endgame_windows)
    scroll_R_tr.grid(column=1, row=11, pady = pady_ef)
    
    # (Row 12) In-track Maneuver
    ttk.Label(engagement_frame, text='In-track threat reaction (m/s):').grid(column=0, row=12, sticky=tk.E)
    c_I_tr_v = tk.DoubleVar()
    c_I_tr_v.set(0)
    scroll_I_tr = tk.Scale(engagement_frame, from_=-20, to = 20, orient ='horizontal', variable = c_I_tr_v, resolution = .25, digits = 4, command = update_endgame_windows)
    scroll_I_tr.grid(column=1, row=12, pady = pady_ef)
    
    # (Row 13) Cross-track Maneuver
    ttk.Label(engagement_frame, text='Cross-track threat reaction (m/s):').grid(column=0, row=13, sticky=tk.E)
    c_C_tr_v = tk.DoubleVar()
    c_C_tr_v.set(0)
    scroll_C_tr = tk.Scale(engagement_frame, from_=-20, to = 20, orient ='horizontal', variable = c_C_tr_v, resolution = .25, digits = 4, command = update_endgame_windows)
    scroll_C_tr.grid(column=1, row=13, pady = pady_ef)
    
    ################## Create the Output Frame ####################################
    #the prefix c_ on a variable represents a contorl variable
    
    padyset = 8
    padxset = 8
    blankholder = "-----------------------------"
    
    output_frame = ttk.Frame(root)
    output_frame.columnconfigure(0, weight=1)
    output_frame.columnconfigure(1, weight=1)
    
    # (Row 0) Trajectory Characteristics
    ttk.Label(output_frame, text='Trajectory Characteristics', font= ('Helvetica 10 underline')).grid(row=0, column = 0, sticky=tk.E, pady=padyset, padx=padxset)
    
    # (Row 1) Epoch Timee
    ttk.Label(output_frame, text='Epoch:').grid(column=0, row=1, sticky=tk.E)
    c_launch_epoch = tk.StringVar()
    c_launch_epoch.set(blankholder)
    ttk.Label(output_frame, textvariable=c_launch_epoch).grid(column=1, row=1, pady=padyset)
    
    # (Row 2) Launch Time
    ttk.Label(output_frame, text='Launch Time (L+0):').grid(column=0, row=2, sticky=tk.E)
    c_launch_time = tk.StringVar()
    c_launch_time.set(blankholder)
    ttk.Label(output_frame, textvariable = c_launch_time).grid(column=1, row=2, pady=padyset)
    
    # (Row 3) Launch Inclination
    ttk.Label(output_frame, text='Launch Inclination:').grid(column=0, row=3, sticky=tk.E)
    c_launch_inc = tk.StringVar()
    c_launch_inc.set(blankholder)
    launch_inc_output = ttk.Label(output_frame, textvariable=c_launch_inc)
    launch_inc_output.grid(row = 3, column = 1, pady=padyset)
    
    # (Row 4) Apogee Altitude
    ttk.Label(output_frame, text='Apogee Altitude:').grid(column=0, row=4, sticky=tk.E)
    c_apogee = tk.StringVar()
    c_apogee.set(blankholder)
    semi_output = ttk.Label(output_frame, textvariable=c_apogee)
    semi_output.grid(row = 4, column = 1, pady=padyset)
    
    # (Row 5) Launch Delta-V (Magnitude)
    ttk.Label(output_frame, text='Launch Delta-V (Magnitude):').grid(column=0, row=5, sticky=tk.E)
    c_launch_sp = tk.StringVar()
    c_launch_sp.set(blankholder)
    launch_sp_output = ttk.Label(output_frame, textvariable=c_launch_sp)
    launch_sp_output.grid(row = 5, column = 1, pady=padyset)
    
    # (Row 6) Earth Assist Velocity (Magnitude)
    ttk.Label(output_frame, text='Earth Delta-V Assist (Magnitude):').grid(column=0, row=6, sticky=tk.E)
    c_earth_sp = tk.StringVar()
    c_earth_sp.set(blankholder)
    earth_sp_output = ttk.Label(output_frame, textvariable=c_earth_sp)
    earth_sp_output.grid(row = 6, column = 1, pady=padyset)
    
    # (Row 7) Impact Velocity (Magnitude)
    ttk.Label(output_frame, text='Impact Velocity (Magnitude):').grid(column=0, row=7, sticky=tk.E)
    c_impact_sp = tk.StringVar()
    c_impact_sp.set(blankholder)
    impact_sp_output = ttk.Label(output_frame, textvariable=c_impact_sp)
    impact_sp_output.grid(row = 7, column = 1, pady=padyset)
    
    # (Row 8) Endgame Duration
    ttk.Label(output_frame, text='Endgame Duration:').grid(row=8, column = 0, sticky=tk.E)
    c_endgame_dt = tk.StringVar()
    c_endgame_dt.set(blankholder)
    endgame_dt_output = ttk.Label(output_frame, textvariable=c_endgame_dt)
    endgame_dt_output.grid(row = 8, column = 1, pady=padyset)
    
    # (Row 9) Endgame Geometry
    ttk.Label(output_frame, text='Endgame Geometry', font= ('Helvetica 10 underline')).grid(row=9, column = 0, sticky=tk.E, pady=padyset)
    
    # (Row 10) KKV Position at Endgame
    ttk.Label(output_frame, text='KKV Endgame Position (RIC):').grid(column=0, row=10, sticky=tk.E)
    c_KKV_r = tk.StringVar()
    c_KKV_r.set(blankholder)
    KKV_r_output = ttk.Label(output_frame, textvariable=c_KKV_r)
    KKV_r_output.grid(row = 10, column = 1, pady=padyset)
    
    # (Row 11) KKV Velocity at Endgame
    ttk.Label(output_frame, text='KKV Endgame Velocity (RIC):').grid(row=11, column = 0, sticky=tk.E)
    c_KKV_v = tk.StringVar()
    c_KKV_v.set(blankholder)
    KKV_v_output = ttk.Label(output_frame, textvariable=c_KKV_v)
    KKV_v_output.grid(row = 11, column = 1, pady=padyset)
    
    # (Row 12) Blue Position
    ttk.Label(output_frame, text='Blue HVA Endgame Position (RIC):').grid(row=12, column = 0, sticky=tk.E)
    c_blue_r = tk.StringVar()
    c_blue_r.set(blankholder)
    blue_r_output = ttk.Label(output_frame, textvariable=c_blue_r)
    blue_r_output.grid(row = 12, column = 1, pady=padyset)
    
    # (Row 13) Blue Velocity
    ttk.Label(output_frame, text='Blue HVA Endgame Velocity (RIC):').grid(row=13, column = 0, sticky=tk.E)
    c_blue_v = tk.StringVar()
    c_blue_v.set(blankholder)
    blue_v_output = ttk.Label(output_frame, textvariable=c_blue_v)
    blue_v_output.grid(row = 13, column = 1, pady = padyset)
    
    # (Row 14) KKV Engagement Constraints
    ttk.Label(output_frame, text='KKV Engagement Constraints', font= ('Helvetica 10 underline')).grid(row=14, column = 0, sticky=tk.E, pady=padyset)
    
    # (Row 15) Blue Threat Reactoin Magnitude
    ttk.Label(output_frame, text='Blue Threat Reactoin (Magnitude):').grid(row=15, column = 0, sticky=tk.E)
    c_tr_mag = tk.StringVar()
    c_tr_mag.set(blankholder)
    tr_mag_output = ttk.Label(output_frame, textvariable=c_tr_mag)
    tr_mag_output.grid(row = 15, column = 1, pady=padyset)
    
    # (Row 16) KKV CTS Angle
    ttk.Label(output_frame, text='KKV CTS Angle:').grid(row=16, column = 0, sticky=tk.E)
    c_CTS_endgame = tk.StringVar()
    c_CTS_endgame.set(blankholder)
    CTS_endgame_output = ttk.Label(output_frame, textvariable=c_CTS_endgame)
    CTS_endgame_output.grid(row = 16, column = 1, pady=padyset)
    
    # (Row 17) KKV Divert Required
    ttk.Label(output_frame, text='KKV Divert Required:').grid(row=17, column = 0, sticky=tk.E)
    c_divert_required = tk.StringVar()
    c_divert_required.set(blankholder)
    divert_required_output = ttk.Label(output_frame, textvariable=c_divert_required)
    divert_required_output.grid(row = 17, column = 1, pady=padyset)
    
    # (Row 18) KKV Look Angle to Target
    ttk.Label(output_frame, text='KKV Look Angle to Target:').grid(row=18, column = 0, sticky=tk.E)
    c_look = tk.StringVar()
    c_look.set(blankholder)
    look_required = ttk.Label(output_frame, textvariable=c_look)
    look_required.grid(row = 18, column = 1, pady=padyset)
    
    # (Row 19) KE WEZ Ellipse Size
    ttk.Label(output_frame, text='KE WEZ Size in RI Plane:').grid(row=19, column = 0, sticky=tk.E)
    c_KE_WEZ_size = tk.StringVar()
    c_KE_WEZ_size.set(blankholder)
    ttk.Label(output_frame, textvariable=c_KE_WEZ_size).grid(row=19, column = 1, pady=padyset)
    
    # (Row 20) FOV Size
    ttk.Label(output_frame, text='FOV Size in RI Plane:').grid(row=20, column = 0, sticky=tk.E)
    c_FOV_size = tk.StringVar()
    c_FOV_size.set(blankholder)
    ttk.Label(output_frame, textvariable = c_FOV_size).grid(row=20, column = 1, pady=padyset)
    
    
    ################## Set the padding for the Graph Frames #######################
    #Set Padding for the graphs
    graphpadx = 20
    graphpady = 20
    size_x = 4
    size_y = 3.5
    
    
    ################## Create the IC Plane ########################################
    #Create Frame for IC Trajectory
    fig_IC = plt.Figure(figsize=(size_x,size_y), dpi=100)
    ax_IC = fig_IC.add_subplot(111)
    chart_type_IC = FigureCanvasTkAgg(fig_IC, root)
    chart_type_IC.get_tk_widget().grid(column = 2, row = 1)
    ax_IC.set_title('Endgame Geometry I-C Plane')
    plt.tight_layout()
    
    ################## Create the RC Plane ########################################
    #Create Frame for RC Trajectory
    fig_RC = plt.Figure(figsize=(size_x,size_y), dpi=100)
    ax_RC = fig_RC.add_subplot(111)
    chart_type_RC = FigureCanvasTkAgg(fig_RC, root)
    chart_type_RC.get_tk_widget().grid(column = 3, row = 1)
    ax_RC.set_title('Endgame Geometry R-C Plane')
    plt.tight_layout()
    
    update_now_global()
    
    #################### Place Engagement and Output Frame ########################
    #place sub frames in the root
    engagement_frame.grid(column=0, row=0, rowspan=2)
    output_frame.grid(column=1, row=0, rowspan=2)
    
    
    ##############################################
    # Execute Main Loop
    root.mainloop()

if __name__ == '__main__':
    main()