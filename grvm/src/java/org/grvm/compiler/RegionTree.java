package org.grvm.compiler;

import java.util.Enumeration;

import org.jikesrvm.compilers.opt.controlflow.LSTGraph;
import org.jikesrvm.compilers.opt.ir.IR;
import org.jikesrvm.compilers.opt.util.GraphNode;

public class RegionTree extends LSTGraph {
  public static void perform(IR ir) {
    ir.HIRInfo.loopStructureTree = new RegionTree(ir, ir.HIRInfo.loopStructureTree);
  }

  public RegionTree(IR ir, LSTGraph graph) {
    super(graph);
    rootNode = new RegionNode(ir, rootNode);
  }

  public void print() {
  	print((RegionNode)rootNode, 0);
  }
  
  private void print(RegionNode n, int depth) {
  	for(Enumeration<GraphNode> e  = n.outNodes();
  			e.hasMoreElements();) {
  		RegionNode c = (RegionNode)e.nextElement();
  		print(c, depth + 1);
  	}
  }
}
