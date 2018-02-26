package org.grvm.compiler;

import org.jikesrvm.compilers.opt.DefUse;
import org.jikesrvm.compilers.opt.OptOptions;
import org.jikesrvm.compilers.opt.controlflow.DominatorsPhase;
import org.jikesrvm.compilers.opt.driver.CompilerPhase;
import org.jikesrvm.compilers.opt.ir.IR;

public class RegionAnalysis extends CompilerPhase {

  @Override
  public final String getName() {
    return "Region Analysis";
  }

  @Override
  public CompilerPhase newExecution(IR ir) {
  	return this;
  }

  @Override
  public boolean shouldPerform(OptOptions options) {
    return options.getOptLevel() >= 0;
  }

  @Override
  public final void perform(IR ir) {
    if (ir.hasReachableExceptionHandlers()) {
    	return;
    } 
    new DominatorsPhase(false).perform(ir);
    DefUse.computeDU(ir);
    RegionTree regionTree = new RegionTree(ir, ir.HIRInfo.loopStructureTree);
    ir.HIRInfo.loopStructureTree = regionTree;
  }
}
