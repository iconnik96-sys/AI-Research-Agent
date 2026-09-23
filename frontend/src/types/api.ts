/**
 * API Type Definitions matching FastAPI Pydantic Models
 */

export type ClaimVerificationStatus =
  | 'SUPPORTED'
  | 'PARTIALLY_SUPPORTED'
  | 'UNSUPPORTED'
  | 'INSUFFICIENT_EVIDENCE';

export interface ResearchSubQuestion {
  id: string;
  question: string;
  search_query: string;
  reason: string;
}

export interface ResearchPlan {
  original_question: string;
  sub_questions: ResearchSubQuestion[];
}

export interface SourceItem {
  title: string;
  url: string;
  content: string;
  score: number | null;
}

export interface Document {
  url: string;
  title: string;
  text: string;
  score: number | null;
  char_count: number;
}

export interface VerifiedClaim {
  id: string;
  claim: string;
  status: ClaimVerificationStatus;
  reason: string;
  evidence_ids: string[];
  supporting_evidence_ids: string[];
}

export interface SourceReference {
  id: string;
  title: string;
  url: string;
}

export interface ResearchSection {
  heading: string;
  content: string;
  citations: string[];
}

export interface ResearchReport {
  title: string;
  summary: string;
  sections: ResearchSection[];
  sources: SourceReference[];
}

export interface ResearchRequest {
  question: string;
}

export interface ResearchResponse {
  session_id: string | null;
  question: string;
  status: string;
  plan: ResearchPlan | null;
  sources: SourceItem[];
  documents: Document[];
  claims: VerifiedClaim[];
  report: ResearchReport | null;
  message: string | null;
}

export interface HealthResponse {
  status: string;
}

export interface ReadyResponse {
  status: string;
  database: string;
}

export interface ApiError {
  message: string;
  status?: number;
  requestId?: string;
  retryAfter?: number;
}
