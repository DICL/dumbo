package com.xiilab.metric.configs;

import javax.sql.DataSource;

import org.apache.ibatis.session.SqlSessionFactory;
import org.mybatis.spring.SqlSessionFactoryBean;
import org.mybatis.spring.SqlSessionTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.Resource;
import org.springframework.jdbc.datasource.SimpleDriverDataSource;

import com.xiilab.metric.model.AmbariYarnAppMonitorConfig;
import com.xiilab.metric.utilities.AmbariProperty;


// timescaleDB - Mybatis 은 ambari-rest api 로 호출해서 나온 설정값으로 접속을 해야하기 때문에 spring config 으로 Bean 설정함
// mybatis-spring 설정 (timescaleDB)
@Configuration
public class MybatisConfig {
	
	@Autowired
	private AmbariProperty ambariProperty;
	
	// DB 접속 설정
	@Bean(name="timescale-dataSource")
	public DataSource timescaleDataSource() {
		SimpleDriverDataSource dataSource = new SimpleDriverDataSource();
		
		// ambari YarnAppMonitor service config 가져오기
		AmbariYarnAppMonitorConfig configurations = ambariProperty.getAmbariYarnAppMonitorConfig();
		
		// postgresql 설정
		dataSource.setDriverClass(org.postgresql.Driver.class);
		dataSource.setUrl(configurations.getTimescaleDB_connet_url());
		dataSource.setUsername(configurations.getTimescaleDB_username());
		dataSource.setPassword(configurations.getTimescaleDB_password());
		
		return dataSource;
	}
	
	// 맵퍼 설정
	@Bean(name="timescale-sqlSessionFactory")
	public SqlSessionFactory timescaleSqlSessionFactory(	) throws Exception {
		
		// timescale config
		DataSource db2DataSource = timescaleDataSource();
		
		SqlSessionFactoryBean sqlSessionFactoryBean = new SqlSessionFactoryBean();
		// mybatis config xml 
		sqlSessionFactoryBean.setConfigLocation(new ClassPathResource("mybatis/mybatis-config.xml"));
		// "classpath:mybatis/mapper/timescale/*.xml"
		// mybatis mapper
		Resource[] mapperLocations = {
				new ClassPathResource("mybatis/mapper/timescale/timescale.xml")
		};
		sqlSessionFactoryBean.setMapperLocations(mapperLocations);
		sqlSessionFactoryBean.setDataSource(db2DataSource);
		
		return  sqlSessionFactoryBean.getObject();
	}
	
	// mybatis 세션설정
	@Bean(name="timescale-sqlSession")
	public SqlSessionTemplate timescaleSqlSession() throws Exception {
		 return new SqlSessionTemplate(timescaleSqlSessionFactory());
	}

}
