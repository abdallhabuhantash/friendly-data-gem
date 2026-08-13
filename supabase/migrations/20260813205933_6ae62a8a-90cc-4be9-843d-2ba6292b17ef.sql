-- Identity allocation is a trusted-service operation only.
REVOKE EXECUTE ON FUNCTION public.allocate_session_subject_number(uuid) FROM anon;
REVOKE EXECUTE ON FUNCTION public.allocate_session_subject_number(uuid) FROM authenticated;
REVOKE EXECUTE ON FUNCTION public.allocate_session_subject_number(uuid) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.allocate_session_subject_number(uuid) TO service_role;