/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.hadoop.yarn.server.resourcemanager.recovery;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.io.IOException;
import java.util.List;

import javax.crypto.SecretKey;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.ha.HAServiceProtocol;
import org.apache.hadoop.ha.HAServiceProtocol.StateChangeRequestInfo;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.security.token.delegation.DelegationKey;
import org.apache.hadoop.service.Service;
import org.apache.hadoop.yarn.api.records.ApplicationAttemptId;
import org.apache.hadoop.yarn.api.records.ApplicationSubmissionContext;
import org.apache.hadoop.yarn.api.records.Container;
import org.apache.hadoop.yarn.api.records.FinalApplicationStatus;
import org.apache.hadoop.yarn.api.records.impl.pb.ApplicationSubmissionContextPBImpl;
import org.apache.hadoop.yarn.api.records.impl.pb.ContainerPBImpl;
import org.apache.hadoop.yarn.conf.HAUtil;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.hadoop.yarn.security.client.RMDelegationTokenIdentifier;
import org.apache.hadoop.yarn.server.records.Version;
import org.apache.hadoop.yarn.server.records.impl.pb.VersionPBImpl;
import org.apache.hadoop.yarn.server.resourcemanager.ResourceManager;
import org.apache.hadoop.yarn.server.resourcemanager.recovery.records.ApplicationAttemptStateData;
import org.apache.hadoop.yarn.server.resourcemanager.recovery.records.ApplicationStateData;
import org.apache.hadoop.yarn.server.resourcemanager.rmapp.RMApp;
import org.apache.hadoop.yarn.server.resourcemanager.rmapp.attempt.AggregateAppResourceUsage;
import org.apache.hadoop.yarn.server.resourcemanager.rmapp.attempt.RMAppAttempt;
import org.apache.hadoop.yarn.server.resourcemanager.rmapp.attempt.RMAppAttemptMetrics;
import org.apache.hadoop.yarn.server.resourcemanager.rmapp.attempt.RMAppAttemptState;
import org.apache.hadoop.yarn.server.resourcemanager.security.ClientToAMTokenSecretManagerInRM;
import org.apache.hadoop.yarn.util.ConverterUtils;
import org.apache.zookeeper.ZooKeeper;
import org.apache.zookeeper.data.Stat;
import org.junit.Assert;
import org.junit.Test;

public class TestZKRMStateStore extends RMStateStoreTestBase {

  public static final Log LOG = LogFactory.getLog(TestZKRMStateStore.class);
  private static final int ZK_TIMEOUT_MS = 1000;

  class TestZKRMStateStoreTester implements RMStateStoreHelper {

    ZooKeeper client;
    TestZKRMStateStoreInternal store;
    String workingZnode;

    class TestZKRMStateStoreInternal extends ZKRMStateStore {

      public TestZKRMStateStoreInternal(Configuration conf, String workingZnode)
          throws Exception {
        init(conf);
        start();
        assertTrue(znodeWorkingPath.equals(workingZnode));
      }

      @Override
      public ZooKeeper getNewZooKeeper() throws IOException {
        return client;
      }

      public String getVersionNode() {
        return znodeWorkingPath + "/" + ROOT_ZNODE_NAME + "/" + VERSION_NODE;
      }

      public Version getCurrentVersion() {
        return CURRENT_VERSION_INFO;
      }

      public String getAppNode(String appId) {
        return workingZnode + "/" + ROOT_ZNODE_NAME + "/" + RM_APP_ROOT + "/"
            + appId;
      }
    }

    public RMStateStore getRMStateStore() throws Exception {
      YarnConfiguration conf = new YarnConfiguration();
      workingZnode = "/jira/issue/3077/rmstore";
      conf.set(YarnConfiguration.RM_ZK_ADDRESS, hostPort);
      conf.set(YarnConfiguration.ZK_RM_STATE_STORE_PARENT_PATH, workingZnode);
      this.client = createClient();
      this.store = new TestZKRMStateStoreInternal(conf, workingZnode);
      return this.store;
    }

    @Override
    public boolean isFinalStateValid() throws Exception {
      List<String> nodes = client.getChildren(store.znodeWorkingPath, false);
      return nodes.size() == 1;
    }

    @Override
    public void writeVersion(Version version) throws Exception {
      client.setData(store.getVersionNode(), ((VersionPBImpl) version)
        .getProto().toByteArray(), -1);
    }

    @Override
    public Version getCurrentVersion() throws Exception {
      return store.getCurrentVersion();
    }

    public boolean appExists(RMApp app) throws Exception {
      Stat node =
          client.exists(store.getAppNode(app.getApplicationId().toString()),
            false);
      return node !=null;
    }
  }

  @Test (timeout = 60000)
  public void testZKRMStateStoreRealZK() throws Exception {
    TestZKRMStateStoreTester zkTester = new TestZKRMStateStoreTester();
    testRMAppStateStore(zkTester);
    testRMDTSecretManagerStateStore(zkTester);
    testCheckVersion(zkTester);
    testEpoch(zkTester);
    testAppDeletion(zkTester);
    testDeleteStore(zkTester);
    testAMRMTokenSecretManagerStateStore(zkTester);
  }

  @Test (timeout = 60000)
  public void testCheckMajorVersionChange() throws Exception {
    TestZKRMStateStoreTester zkTester = new TestZKRMStateStoreTester() {
      Version VERSION_INFO = Version.newInstance(Integer.MAX_VALUE, 0);

      @Override
      public Version getCurrentVersion() throws Exception {
        return VERSION_INFO;
      }

      @Override
      public RMStateStore getRMStateStore() throws Exception {
        YarnConfiguration conf = new YarnConfiguration();
        workingZnode = "/jira/issue/3077/rmstore";
        conf.set(YarnConfiguration.RM_ZK_ADDRESS, hostPort);
        conf.set(YarnConfiguration.ZK_RM_STATE_STORE_PARENT_PATH, workingZnode);
        this.client = createClient();
        this.store = new TestZKRMStateStoreInternal(conf, workingZnode) {
          Version storedVersion = null;

          @Override
          public Version getCurrentVersion() {
            return VERSION_INFO;
          }

          @Override
          protected synchronized Version loadVersion() throws Exception {
            return storedVersion;
          }

          @Override
          protected synchronized void storeVersion() throws Exception {
            storedVersion = VERSION_INFO;
          }
        };
        return this.store;
      }

    };
    // default version
    RMStateStore store = zkTester.getRMStateStore();
    Version defaultVersion = zkTester.getCurrentVersion();
    store.checkVersion();
    Assert.assertEquals(defaultVersion, store.loadVersion());
  }

  private Configuration createHARMConf(
      String rmIds, String rmId, int adminPort) {
    Configuration conf = new YarnConfiguration();
    conf.setBoolean(YarnConfiguration.RM_HA_ENABLED, true);
    conf.set(YarnConfiguration.RM_HA_IDS, rmIds);
    conf.setBoolean(YarnConfiguration.RECOVERY_ENABLED, true);
    conf.set(YarnConfiguration.RM_STORE, ZKRMStateStore.class.getName());
    conf.set(YarnConfiguration.RM_ZK_ADDRESS, hostPort);
    conf.setInt(YarnConfiguration.RM_ZK_TIMEOUT_MS, ZK_TIMEOUT_MS);
    conf.set(YarnConfiguration.RM_HA_ID, rmId);
    conf.set(YarnConfiguration.RM_WEBAPP_ADDRESS, "localhost:0");

    for (String rpcAddress : YarnConfiguration.getServiceAddressConfKeys(conf)) {
      for (String id : HAUtil.getRMHAIds(conf)) {
        conf.set(HAUtil.addSuffix(rpcAddress, id), "localhost:0");
      }
    }
    conf.set(HAUtil.addSuffix(YarnConfiguration.RM_ADMIN_ADDRESS, rmId),
        "localhost:" + adminPort);
    return conf;
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testFencing() throws Exception {
    StateChangeRequestInfo req = new StateChangeRequestInfo(
        HAServiceProtocol.RequestSource.REQUEST_BY_USER);

    Configuration conf1 = createHARMConf("rm1,rm2", "rm1", 1234);
    conf1.setBoolean(YarnConfiguration.AUTO_FAILOVER_ENABLED, false);
    ResourceManager rm1 = new ResourceManager();
    rm1.init(conf1);
    rm1.start();
    rm1.getRMContext().getRMAdminService().transitionToActive(req);
    assertEquals("RM with ZKStore didn't start",
        Service.STATE.STARTED, rm1.getServiceState());
    assertEquals("RM should be Active",
        HAServiceProtocol.HAServiceState.ACTIVE,
        rm1.getRMContext().getRMAdminService().getServiceStatus().getState());

    Configuration conf2 = createHARMConf("rm1,rm2", "rm2", 5678);
    conf2.setBoolean(YarnConfiguration.AUTO_FAILOVER_ENABLED, false);
    ResourceManager rm2 = new ResourceManager();
    rm2.init(conf2);
    rm2.start();
    rm2.getRMContext().getRMAdminService().transitionToActive(req);
    assertEquals("RM with ZKStore didn't start",
        Service.STATE.STARTED, rm2.getServiceState());
    assertEquals("RM should be Active",
        HAServiceProtocol.HAServiceState.ACTIVE,
        rm2.getRMContext().getRMAdminService().getServiceStatus().getState());

    for (int i = 0; i < ZK_TIMEOUT_MS / 50; i++) {
      if (HAServiceProtocol.HAServiceState.ACTIVE ==
          rm1.getRMContext().getRMAdminService().getServiceStatus().getState()) {
        Thread.sleep(100);
      }
    }
    assertEquals("RM should have been fenced",
        HAServiceProtocol.HAServiceState.STANDBY,
        rm1.getRMContext().getRMAdminService().getServiceStatus().getState());
    assertEquals("RM should be Active",
        HAServiceProtocol.HAServiceState.ACTIVE,
        rm2.getRMContext().getRMAdminService().getServiceStatus().getState());
  }
  
  @Test
  public void testFencedState() throws Exception {
    TestZKRMStateStoreTester zkTester = new TestZKRMStateStoreTester();
	RMStateStore store = zkTester.getRMStateStore();
   
    // Move state to FENCED from ACTIVE
    store.updateFencedState();
    assertEquals("RMStateStore should have been in fenced state",
            true, store.isFencedState());    

    long submitTime = System.currentTimeMillis();
    long startTime = submitTime + 1000;

    // Add a new app
    RMApp mockApp = mock(RMApp.class);
    ApplicationSubmissionContext context =
      new ApplicationSubmissionContextPBImpl();
    when(mockApp.getSubmitTime()).thenReturn(submitTime);
    when(mockApp.getStartTime()).thenReturn(startTime);
    when(mockApp.getApplicationSubmissionContext()).thenReturn(context);
    when(mockApp.getUser()).thenReturn("test");
    store.storeNewApplication(mockApp);
    assertEquals("RMStateStore should have been in fenced state",
            true, store.isFencedState());

    // Add a new attempt
    ClientToAMTokenSecretManagerInRM clientToAMTokenMgr =
            new ClientToAMTokenSecretManagerInRM();
    ApplicationAttemptId attemptId = ConverterUtils
            .toApplicationAttemptId("appattempt_1234567894321_0001_000001");
    SecretKey clientTokenMasterKey =
                clientToAMTokenMgr.createMasterKey(attemptId);
    RMAppAttemptMetrics mockRmAppAttemptMetrics = 
         mock(RMAppAttemptMetrics.class);
    Container container = new ContainerPBImpl();
    container.setId(ConverterUtils.toContainerId("container_1234567891234_0001_01_000001"));
    RMAppAttempt mockAttempt = mock(RMAppAttempt.class);
    when(mockAttempt.getAppAttemptId()).thenReturn(attemptId);
    when(mockAttempt.getMasterContainer()).thenReturn(container);
    when(mockAttempt.getClientTokenMasterKey())
        .thenReturn(clientTokenMasterKey);
    when(mockAttempt.getRMAppAttemptMetrics())
        .thenReturn(mockRmAppAttemptMetrics);
    when(mockRmAppAttemptMetrics.getAggregateAppResourceUsage())
        .thenReturn(new AggregateAppResourceUsage(0,0));
    store.storeNewApplicationAttempt(mockAttempt);
    assertEquals("RMStateStore should have been in fenced state",
            true, store.isFencedState());

    long finishTime = submitTime + 1000;
    // Update attempt
    ApplicationAttemptStateData newAttemptState =
      ApplicationAttemptStateData.newInstance(attemptId, container,
            store.getCredentialsFromAppAttempt(mockAttempt),
            startTime, RMAppAttemptState.FINISHED, "testUrl", 
            "test", FinalApplicationStatus.SUCCEEDED, 100, 
            finishTime, 0, 0);
    store.updateApplicationAttemptState(newAttemptState);
    assertEquals("RMStateStore should have been in fenced state",
            true, store.isFencedState());

    // Update app
    ApplicationStateData appState = ApplicationStateData.newInstance(submitTime, 
            startTime, context, "test");
    store.updateApplicationState(appState);
    assertEquals("RMStateStore should have been in fenced state",
            true, store.isFencedState());

    // Remove app
    store.removeApplication(mockApp);
    assertEquals("RMStateStore should have been in fenced state",
            true, store.isFencedState());

    // store RM delegation token;
    RMDelegationTokenIdentifier dtId1 =
        new RMDelegationTokenIdentifier(new Text("owner1"),
            new Text("renewer1"), new Text("realuser1"));
    Long renewDate1 = new Long(System.currentTimeMillis()); 
    dtId1.setSequenceNumber(1111);
    store.storeRMDelegationToken(dtId1, renewDate1);
    assertEquals("RMStateStore should have been in fenced state", true,
        store.isFencedState());

    store.updateRMDelegationToken(dtId1, renewDate1);
    assertEquals("RMStateStore should have been in fenced state", true,
        store.isFencedState());

    // remove delegation key;
    store.removeRMDelegationToken(dtId1);
    assertEquals("RMStateStore should have been in fenced state", true,
        store.isFencedState());

    // store delegation master key;
    DelegationKey key = new DelegationKey(1234, 4321, "keyBytes".getBytes());
    store.storeRMDTMasterKey(key);
    assertEquals("RMStateStore should have been in fenced state", true,
        store.isFencedState());

    // remove delegation master key;
    store.removeRMDTMasterKey(key);
    assertEquals("RMStateStore should have been in fenced state", true,
        store.isFencedState());

    // store or update AMRMToken;
    store.storeOrUpdateAMRMTokenSecretManager(null, false);
    assertEquals("RMStateStore should have been in fenced state", true,
        store.isFencedState());

    store.close();
  }
}
