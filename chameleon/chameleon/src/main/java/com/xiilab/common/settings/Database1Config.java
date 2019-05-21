package com.xiilab.common.settings;

import javax.sql.DataSource;

import org.apache.ibatis.session.SqlSessionFactory;
import org.mybatis.spring.SqlSessionFactoryBean;
import org.mybatis.spring.SqlSessionTemplate;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.jdbc.DataSourceBuilder;
import org.springframework.context.ApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;
import org.springframework.jdbc.datasource.SimpleDriverDataSource;
import org.springframework.transaction.annotation.EnableTransactionManagement;

import com.xiilab.models.AmbariYarnAppMonitorConfig;
import com.xiilab.utilities.AmbariProperty;

// Ambari DataBase Config
@Configuration
@MapperScan(value="com.xiilab.mapper1", sqlSessionFactoryRef="db1SqlSessionFactory")
@EnableTransactionManagement
public class Database1Config {
	
	@Autowired
	private AmbariProperty ambariProperty;
	
//	@Bean(name = "db1DataSource")
//    @Primary
//    @ConfigurationProperties(prefix = "spring.db1.datasource")
//    public DataSource db1DataSource() {
//        return DataSourceBuilder.create().build();
//    }
	
	@Bean(name = "db1DataSource")
	@Primary
	public DataSource db1DataSource() {
		SimpleDriverDataSource dataSource = new SimpleDriverDataSource();
		
		AmbariYarnAppMonitorConfig configurations;
		try {
			configurations = ambariProperty.getAmbariYarnAppMonitorConfig();
			dataSource.setDriverClass(org.postgresql.Driver.class);
			dataSource.setUrl(configurations.getTimescaleDB_connet_url());
			dataSource.setUsername(configurations.getTimescaleDB_username());
			dataSource.setPassword(configurations.getTimescaleDB_password());
		} catch (Exception e) {
			dataSource.setDriverClass(org.postgresql.Driver.class);
			dataSource.setUrl("jdbc:postgresql://192.168.1.190/ambari");
			dataSource.setUsername("postgres");
			dataSource.setPassword("postgres");
			e.printStackTrace();
		}
		
		return dataSource;
	}
 
    @Bean(name = "db1SqlSessionFactory")
    @Primary
    public SqlSessionFactory db1SqlSessionFactory(@Qualifier("db1DataSource") DataSource db1DataSource, ApplicationContext applicationContext) throws Exception {
        SqlSessionFactoryBean sqlSessionFactoryBean = new SqlSessionFactoryBean();
        sqlSessionFactoryBean.setDataSource(db1DataSource);
        sqlSessionFactoryBean.setMapperLocations(applicationContext.getResources("classpath:mybatis/mappers/dao1/*.xml"));
        return sqlSessionFactoryBean.getObject();
    }
 
    @Bean(name = "db1SqlSessionTemplate")
    @Primary
    public SqlSessionTemplate db1SqlSessionTemplate(SqlSessionFactory db1SqlSessionFactory) throws Exception {
 
        return new SqlSessionTemplate(db1SqlSessionFactory);
    }
}
