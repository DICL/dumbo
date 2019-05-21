package com.xiilab.mapper2;

import java.util.List;
import java.util.Map;

public interface AmbariMapper {

	/**
	 * Metric Registry View 에 저장되어 있는 Metric list 가져오기
	 * @return
	 */
	List<Map<String, Object>> getMetricList();

}
