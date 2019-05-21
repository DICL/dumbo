-- 미사용 스크립트
CREATE SEQUENCE ambari.chameleon_data_num_seq
    INCREMENT 1
    START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1;

ALTER SEQUENCE ambari.chameleon_data_num_seq
    OWNER TO ambari;



-- Table: ambari.chameleon_data

-- DROP TABLE ambari.chameleon_data;

-- yarn job resource 저장

CREATE TABLE ambari.chameleon_data
(
    num integer NOT NULL DEFAULT nextval('chameleon_data_num_seq'::regclass),   /* 기본키 (자동증가)*/
    val character varying(4096) COLLATE pg_catalog."default",                   /* 데이터 (host 정보, application app id, pid, cpu,mem 등) */
    create_date timestamp without time zone,                                    /* 생성일자 */
    rowkey character varying(255) COLLATE pg_catalog."default",                 /* rowkey (<UNIXTIME>_<PID>)*/
    metric character varying(255) COLLATE pg_catalog."default",                 /* 메트릭정보 (node,application_id,%MEM,%CPU,PID)*/
    pid character varying(255) COLLATE pg_catalog."default",                    /* PID 정보 */
    CONSTRAINT chameleon_data_pkey PRIMARY KEY (num)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE ambari.chameleon_data
    OWNER to ambari;




CREATE SEQUENCE ambari.application_app_list_num_seq
    INCREMENT 1
    START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1;

ALTER SEQUENCE ambari.application_app_list_num_seq
    OWNER TO ambari;





-- Table: ambari.application_app_list

-- DROP TABLE ambari.application_app_list;

-- 얀 잡 리스트 api 정보저장

/*
    
*/

CREATE TABLE ambari.application_app_list
(
    num integer NOT NULL DEFAULT nextval('application_app_list_num_seq'::regclass), /* 기본키 (자동증가)*/
    aid character varying(255) COLLATE pg_catalog."default",                        /* application app id */
    pid character varying(255) COLLATE pg_catalog."default",                        /* PID */
    state character varying(255) COLLATE pg_catalog."default",                      /* 상태 */
    stime character varying(255) COLLATE pg_catalog."default",                      /* 시작 시간 */
    etime character varying(255) COLLATE pg_catalog."default",                      /* 종료 시간 */
    node character varying(255) COLLATE pg_catalog."default",                       /* 호스트 이름 */
    cid character varying(255) COLLATE pg_catalog."default",                        /* container_id */
    CONSTRAINT application_app_list_pkey PRIMARY KEY (num)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE ambari.application_app_list
    OWNER to ambari;