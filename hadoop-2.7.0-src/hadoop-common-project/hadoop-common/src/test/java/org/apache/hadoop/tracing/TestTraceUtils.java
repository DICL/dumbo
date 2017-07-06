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
package org.apache.hadoop.tracing;

import static org.junit.Assert.assertEquals;
import java.util.LinkedList;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.tracing.SpanReceiverInfo.ConfigurationPair;
import org.apache.htrace.HTraceConfiguration;
import org.junit.Test;

public class TestTraceUtils {
  @Test
  public void testWrappedHadoopConf() {
    String key = "sampler";
    String value = "ProbabilitySampler";
    Configuration conf = new Configuration();
    conf.set(TraceUtils.HTRACE_CONF_PREFIX + key, value);
    HTraceConfiguration wrapped = TraceUtils.wrapHadoopConf(conf);
    assertEquals(value, wrapped.get(key));
  }

  @Test
  public void testExtraConfig() {
    String key = "test.extra.config";
    String oldValue = "old value";
    String newValue = "new value";
    Configuration conf = new Configuration();
    conf.set(TraceUtils.HTRACE_CONF_PREFIX + key, oldValue);
    LinkedList<ConfigurationPair> extraConfig =
        new LinkedList<ConfigurationPair>();
    extraConfig.add(new ConfigurationPair(key, newValue));
    HTraceConfiguration wrapped = TraceUtils.wrapHadoopConf(conf, extraConfig);
    assertEquals(newValue, wrapped.get(key));
  }
}
