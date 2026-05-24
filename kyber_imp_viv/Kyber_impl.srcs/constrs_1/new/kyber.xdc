# ============================================================================
# Kyber SCA Implementation - Arty A7-100 (xc7a100tcsg324-1)
# ============================================================================

# ============================================================================
# CLOCK CONSTRAINT (100 MHz system clock)
# ============================================================================
set_property -dict { PACKAGE_PIN E3 IOSTANDARD LVCMOS33 } [get_ports clk]
create_clock -add -name sys_clk_pin -period 10.00 -waveform {0 5} [get_ports clk]

# ============================================================================
# RESET SIGNAL
# ============================================================================
set_property -dict { PACKAGE_PIN D9 IOSTANDARD LVCMOS33 } [get_ports rst_n]

# ============================================================================
# DATA INPUT (din[7:0])
# Using PMOD JA connector for input
# ============================================================================
set_property -dict { PACKAGE_PIN C17 IOSTANDARD LVCMOS33 } [get_ports {din[0]}]
set_property -dict { PACKAGE_PIN D18 IOSTANDARD LVCMOS33 } [get_ports {din[1]}]
set_property -dict { PACKAGE_PIN E18 IOSTANDARD LVCMOS33 } [get_ports {din[2]}]
set_property -dict { PACKAGE_PIN G17 IOSTANDARD LVCMOS33 } [get_ports {din[3]}]
set_property -dict { PACKAGE_PIN D17 IOSTANDARD LVCMOS33 } [get_ports {din[4]}]
set_property -dict { PACKAGE_PIN E17 IOSTANDARD LVCMOS33 } [get_ports {din[5]}]
set_property -dict { PACKAGE_PIN F18 IOSTANDARD LVCMOS33 } [get_ports {din[6]}]
set_property -dict { PACKAGE_PIN G18 IOSTANDARD LVCMOS33 } [get_ports {din[7]}]

# ============================================================================
# DATA OUTPUT (dout[7:0])
# Using PMOD JB connector for output
# ============================================================================
set_property -dict { PACKAGE_PIN D14 IOSTANDARD LVCMOS33 } [get_ports {dout[0]}]
set_property -dict { PACKAGE_PIN F16 IOSTANDARD LVCMOS33 } [get_ports {dout[1]}]
set_property -dict { PACKAGE_PIN G16 IOSTANDARD LVCMOS33 } [get_ports {dout[2]}]
set_property -dict { PACKAGE_PIN H14 IOSTANDARD LVCMOS33 } [get_ports {dout[3]}]
set_property -dict { PACKAGE_PIN E16 IOSTANDARD LVCMOS33 } [get_ports {dout[4]}]
set_property -dict { PACKAGE_PIN F13 IOSTANDARD LVCMOS33 } [get_ports {dout[5]}]
set_property -dict { PACKAGE_PIN G13 IOSTANDARD LVCMOS33 } [get_ports {dout[6]}]
set_property -dict { PACKAGE_PIN H16 IOSTANDARD LVCMOS33 } [get_ports {dout[7]}]

# ============================================================================
# CONTROL SIGNALS
# Using remaining PMOD JC pins
# ============================================================================
set_property -dict { PACKAGE_PIN K1 IOSTANDARD LVCMOS33 } [get_ports din_valid]
set_property -dict { PACKAGE_PIN F6 IOSTANDARD LVCMOS33 } [get_ports dout_valid]
set_property -dict { PACKAGE_PIN J2 IOSTANDARD LVCMOS33 } [get_ports busy]

# ============================================================================
# OPTIONAL: Cycle counter output (for triggering scope)
# Using onboard LEDs for monitoring
# ============================================================================
set_property -dict { PACKAGE_PIN H17 IOSTANDARD LVCMOS33 } [get_ports {cycle_counter[0]}]
set_property -dict { PACKAGE_PIN K15 IOSTANDARD LVCMOS33 } [get_ports {cycle_counter[1]}]
set_property -dict { PACKAGE_PIN J13 IOSTANDARD LVCMOS33 } [get_ports {cycle_counter[2]}]
set_property -dict { PACKAGE_PIN N14 IOSTANDARD LVCMOS33 } [get_ports {cycle_counter[3]}]

# ============================================================================
# ADDRESS LINES (Optional - if using external memory)
# Using remaining PMOD pins
# ============================================================================
set_property -dict { PACKAGE_PIN H4 IOSTANDARD LVCMOS33 } [get_ports {addr[0]}]
set_property -dict { PACKAGE_PIN J4 IOSTANDARD LVCMOS33 } [get_ports {addr[1]}]
set_property -dict { PACKAGE_PIN G6 IOSTANDARD LVCMOS33 } [get_ports {addr[2]}]
set_property -dict { PACKAGE_PIN H6 IOSTANDARD LVCMOS33 } [get_ports {addr[3]}]
set_property -dict { PACKAGE_PIN K3 IOSTANDARD LVCMOS33 } [get_ports {addr[4]}]
set_property -dict { PACKAGE_PIN K2 IOSTANDARD LVCMOS33 } [get_ports {addr[5]}]

# ============================================================================
# TIMING CONSTRAINTS
# ============================================================================
set_input_delay -clock sys_clk_pin -min 0.5 [get_ports {din* din_valid addr*}]
set_input_delay -clock sys_clk_pin -max 2.0 [get_ports {din* din_valid addr*}]

set_output_delay -clock sys_clk_pin -min 0.5 [get_ports {dout* dout_valid busy cycle_counter*}]
set_output_delay -clock sys_clk_pin -max 2.0 [get_ports {dout* dout_valid busy cycle_counter*}]

# ============================================================================
# DESIGN PRESERVATION (Critical for SCA Analysis)
# ============================================================================
# Keep hierarchy to prevent optimization that would hide power leakage
set_property KEEP_HIERARCHY TRUE [get_cells kyber_core]

# Keep intermediate signals (they leak power even if not used functionally)
set_property KEEP TRUE [get_nets {busy dout_valid cycle_counter*}]

# ============================================================================
# PLACEMENT & ROUTING STRATEGY
# ============================================================================
# Avoid aggressive optimizations that remove side-channel information
set_property strategy Performance_ExploreWithRuntimeLimit [get_runs impl_1]

# Allow flexible clock routing to avoid placement conflicts
set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets clk_IBUF]

# ============================================================================
# POWER MEASUREMENT CONFIGURATION
# ============================================================================
# For ChipWhisperer or oscilloscope-based side-channel analysis:
# 1. Connect shunt resistor (0.5? - 1?) in series with FPGA power supply
# 2. Probe voltage across shunt with oscilloscope or ADC
# 3. Use cycle_counter outputs to trigger scope capture
# 4. Clock edge (E3) can be used as external trigger for scope

# Ensure no global buffers interfere with power measurement
# (Global clock is acceptable as it affects all paths equally)

# ============================================================================
# NOTES FOR SIDE-CHANNEL ANALYSIS
# ============================================================================
# Power measurement points on Arty A7-100:
# - VCCO (I/O supply) - 3.3V
# - VCCINT (core supply) - 1.0V
# - VCCBRAM (BRAM supply) - 1.0V
#
# For best SCA results, measure VCCINT with shunt resistor
# Typical shunt: 1? resistor in series
# Oscilloscope settings: 50mV/div, 1GS/s sampling
#
# Trigger options:
# - Rising edge of cycle_counter[3] (LSB change)
# - External trigger from control PC via GPIO
# - Internal trigger on din_valid
