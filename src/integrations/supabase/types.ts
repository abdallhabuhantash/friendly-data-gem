export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  // Allows to automatically instantiate createClient with right options
  // instead of createClient<Database, { PostgrestVersion: 'XX' }>(URL, KEY)
  __InternalSupabase: {
    PostgrestVersion: "14.15"
  }
  public: {
    Tables: {
      ai_rule_cameras: {
        Row: {
          camera_id: string
          rule_id: string
        }
        Insert: {
          camera_id: string
          rule_id: string
        }
        Update: {
          camera_id?: string
          rule_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "ai_rule_cameras_camera_id_fkey"
            columns: ["camera_id"]
            isOneToOne: false
            referencedRelation: "cameras"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "ai_rule_cameras_rule_id_fkey"
            columns: ["rule_id"]
            isOneToOne: false
            referencedRelation: "ai_rules"
            referencedColumns: ["id"]
          },
        ]
      }
      ai_rules: {
        Row: {
          association_confidence_threshold: number
          available: boolean
          confidence_threshold: number
          cooldown_seconds: number
          created_at: string
          description: string
          enabled: boolean
          engine_key: string | null
          id: string
          instant_confidence_threshold: number
          instant_detection_enabled: boolean
          min_duration_seconds: number
          min_matching_frames: number
          name: string
          person_confidence_threshold: number
          require_person_association: boolean
          save_snapshot: boolean
          severity: string
          sound_notification: boolean
        }
        Insert: {
          association_confidence_threshold?: number
          available?: boolean
          confidence_threshold?: number
          cooldown_seconds?: number
          created_at?: string
          description?: string
          enabled?: boolean
          engine_key?: string | null
          id?: string
          instant_confidence_threshold?: number
          instant_detection_enabled?: boolean
          min_duration_seconds?: number
          min_matching_frames?: number
          name: string
          person_confidence_threshold?: number
          require_person_association?: boolean
          save_snapshot?: boolean
          severity?: string
          sound_notification?: boolean
        }
        Update: {
          association_confidence_threshold?: number
          available?: boolean
          confidence_threshold?: number
          cooldown_seconds?: number
          created_at?: string
          description?: string
          enabled?: boolean
          engine_key?: string | null
          id?: string
          instant_confidence_threshold?: number
          instant_detection_enabled?: boolean
          min_duration_seconds?: number
          min_matching_frames?: number
          name?: string
          person_confidence_threshold?: number
          require_person_association?: boolean
          save_snapshot?: boolean
          severity?: string
          sound_notification?: boolean
        }
        Relationships: []
      }
      camera_credentials: {
        Row: {
          camera_id: string
          password: string | null
          rtsp_url: string
          updated_at: string
          username: string | null
        }
        Insert: {
          camera_id: string
          password?: string | null
          rtsp_url: string
          updated_at?: string
          username?: string | null
        }
        Update: {
          camera_id?: string
          password?: string | null
          rtsp_url?: string
          updated_at?: string
          username?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "camera_credentials_camera_id_fkey"
            columns: ["camera_id"]
            isOneToOne: true
            referencedRelation: "cameras"
            referencedColumns: ["id"]
          },
        ]
      }
      cameras: {
        Row: {
          active: boolean
          ai_enabled: boolean
          channel: number
          created_at: string
          fps: number
          host: string
          id: string
          is_demo: boolean
          last_heartbeat_at: string
          location: string
          name: string
          recording: boolean
          resolution: string
          rtsp_port: number
          source_type: string
          status: string
          stream_path: string
          stream_profile: string
          updated_at: string
        }
        Insert: {
          active?: boolean
          ai_enabled?: boolean
          channel?: number
          created_at?: string
          fps?: number
          host?: string
          id?: string
          is_demo?: boolean
          last_heartbeat_at?: string
          location?: string
          name: string
          recording?: boolean
          resolution?: string
          rtsp_port?: number
          source_type?: string
          status?: string
          stream_path?: string
          stream_profile?: string
          updated_at?: string
        }
        Update: {
          active?: boolean
          ai_enabled?: boolean
          channel?: number
          created_at?: string
          fps?: number
          host?: string
          id?: string
          is_demo?: boolean
          last_heartbeat_at?: string
          location?: string
          name?: string
          recording?: boolean
          resolution?: string
          rtsp_port?: number
          source_type?: string
          status?: string
          stream_path?: string
          stream_profile?: string
          updated_at?: string
        }
        Relationships: []
      }
      events: {
        Row: {
          association_confidence: number | null
          association_status: string
          camera_id: string | null
          camera_name: string
          confidence: number
          created_at: string
          detected_at: string
          detection_duration_seconds: number | null
          detection_frame_count: number | null
          duration_seconds: number
          evidence: Json
          id: string
          note: string | null
          person_tracking_id: string | null
          reviewed_at: string | null
          reviewed_by: string | null
          rule_id: string | null
          severity: string
          snapshot_path: string | null
          source_mode: string
          status: string
          trigger_confidence: number | null
          trigger_object_class: string | null
          type: string
        }
        Insert: {
          association_confidence?: number | null
          association_status?: string
          camera_id?: string | null
          camera_name?: string
          confidence?: number
          created_at?: string
          detected_at?: string
          detection_duration_seconds?: number | null
          detection_frame_count?: number | null
          duration_seconds?: number
          evidence?: Json
          id?: string
          note?: string | null
          person_tracking_id?: string | null
          reviewed_at?: string | null
          reviewed_by?: string | null
          rule_id?: string | null
          severity?: string
          snapshot_path?: string | null
          source_mode?: string
          status?: string
          trigger_confidence?: number | null
          trigger_object_class?: string | null
          type: string
        }
        Update: {
          association_confidence?: number | null
          association_status?: string
          camera_id?: string | null
          camera_name?: string
          confidence?: number
          created_at?: string
          detected_at?: string
          detection_duration_seconds?: number | null
          detection_frame_count?: number | null
          duration_seconds?: number
          evidence?: Json
          id?: string
          note?: string | null
          person_tracking_id?: string | null
          reviewed_at?: string | null
          reviewed_by?: string | null
          rule_id?: string | null
          severity?: string
          snapshot_path?: string | null
          source_mode?: string
          status?: string
          trigger_confidence?: number | null
          trigger_object_class?: string | null
          type?: string
        }
        Relationships: [
          {
            foreignKeyName: "events_camera_id_fkey"
            columns: ["camera_id"]
            isOneToOne: false
            referencedRelation: "cameras"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "events_rule_id_fkey"
            columns: ["rule_id"]
            isOneToOne: false
            referencedRelation: "ai_rules"
            referencedColumns: ["id"]
          },
        ]
      }
      exam_invigilators: {
        Row: {
          created_at: string
          exam_session_id: string
          full_name: string
          id: string
          role: string
          updated_at: string
        }
        Insert: {
          created_at?: string
          exam_session_id: string
          full_name: string
          id?: string
          role?: string
          updated_at?: string
        }
        Update: {
          created_at?: string
          exam_session_id?: string
          full_name?: string
          id?: string
          role?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "exam_invigilators_exam_session_id_fkey"
            columns: ["exam_session_id"]
            isOneToOne: false
            referencedRelation: "exam_sessions"
            referencedColumns: ["id"]
          },
        ]
      }
      exam_roster_students: {
        Row: {
          created_at: string
          exam_session_id: string
          full_name: string
          id: string
          university_id: string
          updated_at: string
        }
        Insert: {
          created_at?: string
          exam_session_id: string
          full_name: string
          id?: string
          university_id: string
          updated_at?: string
        }
        Update: {
          created_at?: string
          exam_session_id?: string
          full_name?: string
          id?: string
          university_id?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "exam_roster_students_exam_session_id_fkey"
            columns: ["exam_session_id"]
            isOneToOne: false
            referencedRelation: "exam_sessions"
            referencedColumns: ["id"]
          },
        ]
      }
      exam_session_cameras: {
        Row: {
          camera_id: string
          created_at: string
          exam_session_id: string
          is_primary: boolean
        }
        Insert: {
          camera_id: string
          created_at?: string
          exam_session_id: string
          is_primary?: boolean
        }
        Update: {
          camera_id?: string
          created_at?: string
          exam_session_id?: string
          is_primary?: boolean
        }
        Relationships: [
          {
            foreignKeyName: "exam_session_cameras_camera_id_fkey"
            columns: ["camera_id"]
            isOneToOne: false
            referencedRelation: "cameras"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "exam_session_cameras_exam_session_id_fkey"
            columns: ["exam_session_id"]
            isOneToOne: false
            referencedRelation: "exam_sessions"
            referencedColumns: ["id"]
          },
        ]
      }
      exam_sessions: {
        Row: {
          course_code: string
          created_at: string
          created_by: string | null
          ended_at: string | null
          id: string
          location_label: string
          scheduled_at: string | null
          started_at: string | null
          status: string
          title: string
          updated_at: string
        }
        Insert: {
          course_code?: string
          created_at?: string
          created_by?: string | null
          ended_at?: string | null
          id?: string
          location_label?: string
          scheduled_at?: string | null
          started_at?: string | null
          status?: string
          title: string
          updated_at?: string
        }
        Update: {
          course_code?: string
          created_at?: string
          created_by?: string | null
          ended_at?: string | null
          id?: string
          location_label?: string
          scheduled_at?: string | null
          started_at?: string | null
          status?: string
          title?: string
          updated_at?: string
        }
        Relationships: []
      }
      profiles: {
        Row: {
          created_at: string
          email: string
          full_name: string
          id: string
          last_active_at: string
          status: string
        }
        Insert: {
          created_at?: string
          email?: string
          full_name?: string
          id: string
          last_active_at?: string
          status?: string
        }
        Update: {
          created_at?: string
          email?: string
          full_name?: string
          id?: string
          last_active_at?: string
          status?: string
        }
        Relationships: []
      }
      service_health: {
        Row: {
          is_demo: boolean
          online: boolean
          payload: Json
          service: string
          updated_at: string
        }
        Insert: {
          is_demo?: boolean
          online?: boolean
          payload?: Json
          service: string
          updated_at?: string
        }
        Update: {
          is_demo?: boolean
          online?: boolean
          payload?: Json
          service?: string
          updated_at?: string
        }
        Relationships: []
      }
      system_settings: {
        Row: {
          ai_service_url: string
          auto_acknowledge_minutes: number
          id: boolean
          operation_mode: string
          retention_days: number
          snapshot_storage: string
          sound_alerts: boolean
          timezone: string
          updated_at: string
          websocket_url: string
        }
        Insert: {
          ai_service_url?: string
          auto_acknowledge_minutes?: number
          id?: boolean
          operation_mode?: string
          retention_days?: number
          snapshot_storage?: string
          sound_alerts?: boolean
          timezone?: string
          updated_at?: string
          websocket_url?: string
        }
        Update: {
          ai_service_url?: string
          auto_acknowledge_minutes?: number
          id?: boolean
          operation_mode?: string
          retention_days?: number
          snapshot_storage?: string
          sound_alerts?: boolean
          timezone?: string
          updated_at?: string
          websocket_url?: string
        }
        Relationships: []
      }
      user_roles: {
        Row: {
          created_at: string
          id: string
          role: Database["public"]["Enums"]["app_role"]
          user_id: string
        }
        Insert: {
          created_at?: string
          id?: string
          role: Database["public"]["Enums"]["app_role"]
          user_id: string
        }
        Update: {
          created_at?: string
          id?: string
          role?: Database["public"]["Enums"]["app_role"]
          user_id?: string
        }
        Relationships: []
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      has_any_role: { Args: { _user_id: string }; Returns: boolean }
      has_role: {
        Args: {
          _role: Database["public"]["Enums"]["app_role"]
          _user_id: string
        }
        Returns: boolean
      }
      is_admin: { Args: never; Returns: boolean }
      review_event: {
        Args: { _event_id: string; _note?: string; _status: string }
        Returns: undefined
      }
    }
    Enums: {
      app_role: "administrator" | "operator"
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DatabaseWithoutInternals = Omit<Database, "__InternalSupabase">

type DefaultSchema = DatabaseWithoutInternals[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I
      }
      ? I
      : never
    : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U
      }
      ? U
      : never
    : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof DatabaseWithoutInternals },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never,
> = DefaultSchemaEnumNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof DatabaseWithoutInternals },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never,
> = PublicCompositeTypeNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never

export const Constants = {
  public: {
    Enums: {
      app_role: ["administrator", "operator"],
    },
  },
} as const
