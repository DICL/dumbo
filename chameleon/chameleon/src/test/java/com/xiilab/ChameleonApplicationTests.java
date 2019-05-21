package com.xiilab;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit4.SpringRunner;

import com.xiilab.ambari.AmbariStatusDAO;
import com.xiilab.mapper1.ApplicationMonitorMapper;
import com.xiilab.mapper1.Db1Mapper;
import com.xiilab.mapper2.AmbariMapper;
import com.xiilab.mapper2.Db2Mapper;
import com.xiilab.models.YarnAppMonitorVO;


@RunWith(SpringRunner.class)
@SpringBootTest
public class ChameleonApplicationTests {

	@Autowired
    Db1Mapper db1Mapper;
	
	@Autowired
	ApplicationMonitorMapper applicationMonitor; // timescaleDB
	
	@Autowired
	AmbariMapper ambariMapper; // ambari postgreSQL
	
	@Autowired
    Db2Mapper db2Mapper; // TEST
	
	@Autowired
	AmbariStatusDAO statusDAO;
	
	@Test
	public void db1Test() throws Exception {
		System.out.println(db1Mapper.getDb1test());
	}
	
	@Test
	public void applicationMonitorTest() {
		System.out.println(applicationMonitor.getContainerIdForApplicationMonitor("application_1536191979328_0005"));
	}
	
	@Test
	public void rowkey_applicationMonitorTest() {
		System.out.println(applicationMonitor.getRowKeyForApplicationMonitor("container_e03_1536191979328_0005_01_000001"));
	}

//	@Test
//	public void applicationMonitorListTest() {
//		YarnAppMonitorVO yarnAppMonitorVO = new YarnAppMonitorVO();
//		System.out.println(applicationMonitor.getYarnAppMonitorList(yarnAppMonitorVO).size());
//	}
	
	@Test
	public void db2Test() throws Exception {
		System.out.println(db2Mapper.getDb2test());
	}

//	@Test
//	public void AmbariClusterTest() throws Exception {
//		System.out.println(statusDAO.getAmbariServerClustreName("supercom_test"));
//	}
}
