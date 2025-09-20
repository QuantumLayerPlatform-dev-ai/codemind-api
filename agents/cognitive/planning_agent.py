"""
Planning Agent for CodeMind Cognitive Software Factory
=====================================================

The Planning Agent is responsible for:
- Understanding business intent from natural language descriptions
- Identifying industry patterns and business models
- Extracting technical requirements and constraints
- Creating comprehensive project specifications
- Setting up the foundation for all subsequent agents
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

try:
    from ...core.logging import get_logger
    from ...services.llm_router import LLMRouter, TaskType
    from ..core.base_agent import BaseAgent
    from ..core.context_manager import (
        SharedContext, Decision, Constraint, DecisionType, ConstraintType
    )
    from ..core.fingerprinting import AgentType, AgentCapability
    from ..core.registry import register_agent
except ImportError:
    from core.logging import get_logger
    from services.llm_router import LLMRouter, TaskType
    from agents.core.base_agent import BaseAgent
    from agents.core.context_manager import (
        SharedContext, Decision, Constraint, DecisionType, ConstraintType
    )
    from agents.core.fingerprinting import AgentType, AgentCapability
    from agents.core.registry import register_agent

logger = get_logger("planning_agent")


@register_agent(AgentType.PLANNING, version="1.0.0")
class PlanningAgent(BaseAgent):
    """
    Planning Agent that transforms business descriptions into technical specifications.

    This agent serves as the entry point for the cognitive software factory,
    understanding business intent and creating the foundation for all subsequent development.
    """

    # Define required inputs and outputs for validation
    REQUIRED_INPUTS = ["business_description"]
    REQUIRED_OUTPUTS = ["technical_specifications", "business_analysis", "feature_list", "constraints"]

    def __init__(self, context: SharedContext, **kwargs):
        """Initialize Planning Agent with business understanding capabilities"""
        super().__init__(
            context=context,
            agent_type=AgentType.PLANNING,
            version="1.0.0",
            **kwargs
        )

        # Add planning-specific capabilities
        self.fingerprint.add_capability(
            AgentCapability(
                name="business_intent_analysis",
                description="Analyze and understand business intent from natural language",
                version="1.0",
                supported_inputs=["business_description", "user_requirements"],
                supported_outputs=["business_model", "industry_classification", "feature_requirements"]
            )
        )

        self.fingerprint.add_capability(
            AgentCapability(
                name="technical_specification_generation",
                description="Generate technical specifications from business requirements",
                version="1.0",
                supported_inputs=["business_analysis", "constraints"],
                supported_outputs=["technical_specifications", "architecture_recommendations"]
            )
        )

        self.fingerprint.add_capability(
            AgentCapability(
                name="constraint_identification",
                description="Identify business, technical, and regulatory constraints",
                version="1.0",
                supported_inputs=["business_context", "industry_classification"],
                supported_outputs=["constraint_list", "compliance_requirements"]
            )
        )

        # Industry knowledge base
        self.industry_patterns = self._load_industry_patterns()
        self.business_models = self._load_business_models()
        self.compliance_rules = self._load_compliance_rules()

    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the planning process.

        Steps:
        1. Analyze business intent and classify industry
        2. Extract feature requirements
        3. Identify constraints and compliance needs
        4. Generate technical specifications
        5. Create project roadmap
        """
        business_description = kwargs.get("business_description") or self.context.original_request
        complexity = kwargs.get("complexity", self.context.complexity_score)

        logger.info(f"Planning agent analyzing: {business_description[:100]}...")

        # Step 1: Business Intent Analysis
        business_analysis = await self._analyze_business_intent(business_description, complexity)

        # Step 2: Feature Extraction
        feature_list = await self._extract_features(business_description, business_analysis)

        # Step 3: Constraint Identification
        constraints = await self._identify_constraints(business_analysis, feature_list)

        # Step 4: Technical Specification Generation
        technical_specs = await self._generate_technical_specifications(
            business_analysis, feature_list, constraints
        )

        # Step 5: Project Roadmap
        roadmap = await self._create_project_roadmap(technical_specs, complexity)

        # Update business intent in context
        self.context.business_intent.description = business_description
        self.context.business_intent.industry = business_analysis.get("industry", "")
        self.context.business_intent.business_model = business_analysis.get("business_model", "")
        self.context.business_intent.key_features = feature_list
        self.context.business_intent.compliance_requirements = [
            c["description"] for c in constraints if c["type"] == "regulatory"
        ]

        # Add decisions made by the planning agent
        decisions = [
            Decision(
                agent_id=self.fingerprint.agent_id,
                agent_type=self.fingerprint.agent_type,
                decision_type=DecisionType.BUSINESS_MODEL,
                description=f"Identified business model: {business_analysis.get('business_model')}",
                rationale=business_analysis.get("business_model_rationale", ""),
                confidence=business_analysis.get("confidence", 0.8)
            ),
            Decision(
                agent_id=self.fingerprint.agent_id,
                agent_type=self.fingerprint.agent_type,
                decision_type=DecisionType.FEATURE_SELECTION,
                description="Selected core features based on business analysis",
                rationale="Features identified through business intent analysis and industry patterns",
                confidence=0.9,
                alternatives_considered=business_analysis.get("alternative_features", [])
            )
        ]

        # Create constraints from analysis
        identified_constraints = []
        for constraint_data in constraints:
            constraint = Constraint(
                constraint_type=ConstraintType(constraint_data["type"]),
                description=constraint_data["description"],
                priority=constraint_data.get("priority", 2),
                source="planning_agent",
                validation_rule=constraint_data.get("validation_rule", "")
            )
            identified_constraints.append(constraint)

        return {
            "business_analysis": business_analysis,
            "feature_list": feature_list,
            "technical_specifications": technical_specs,
            "project_roadmap": roadmap,
            "constraints": [c.__dict__ for c in identified_constraints],
            "decisions": [d.__dict__ for d in decisions],
            "planning_metadata": {
                "complexity_assessment": complexity,
                "industry_classification": business_analysis.get("industry"),
                "confidence_score": business_analysis.get("confidence", 0.8),
                "estimated_development_time": roadmap.get("estimated_duration_weeks"),
                "estimated_cost": roadmap.get("estimated_cost_usd")
            }
        }

    async def _analyze_business_intent(self, description: str, complexity: float) -> Dict[str, Any]:
        """Analyze business intent using LLM with industry knowledge"""

        prompt = f"""
        Analyze the following business description and provide a comprehensive business analysis:

        Business Description: "{description}"
        Complexity Level: {complexity} (0.0 = simple, 1.0 = complex)

        Please analyze and provide:

        1. Industry Classification (choose the most relevant):
           - E-commerce & Retail
           - Healthcare & Medical
           - Financial Services
           - Education & Training
           - Real Estate
           - Food & Restaurant
           - Travel & Tourism
           - Entertainment & Media
           - Business Services
           - Technology & Software
           - Other (specify)

        2. Business Model Type:
           - B2C (Business to Consumer)
           - B2B (Business to Business)
           - B2B2C (Business to Business to Consumer)
           - Marketplace/Platform
           - SaaS (Software as a Service)
           - E-commerce
           - Subscription
           - Freemium
           - Other (specify)

        3. Core Value Proposition:
           - What problem does this solve?
           - Who is the target audience?
           - What makes this unique?

        4. Revenue Model:
           - How will this make money?
           - Pricing strategy
           - Revenue streams

        5. Key Success Metrics:
           - What metrics will measure success?
           - User engagement indicators
           - Business KPIs

        6. Competitive Landscape:
           - Who are the main competitors?
           - What's the competitive advantage?
           - Market positioning

        7. Technical Complexity Assessment:
           - Simple (basic CRUD, simple UI)
           - Medium (real-time features, integrations)
           - Complex (AI/ML, advanced algorithms, scale)

        Return your analysis as a JSON object with clear, actionable insights.
        """

        try:
            # Use LLM router for business intent analysis
            response = await self.llm_router.route_request(
                prompt=prompt,
                task_type=TaskType.BUSINESS_INTENT,
                complexity=complexity,
                max_tokens=2000
            )

            # Parse the JSON response
            analysis_text = response["content"]

            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                try:
                    analysis = json.loads(json_match.group())
                except json.JSONDecodeError:
                    # Fallback to structured parsing
                    analysis = self._parse_business_analysis_fallback(analysis_text)
            else:
                analysis = self._parse_business_analysis_fallback(analysis_text)

            # Add metadata
            analysis.update({
                "confidence": response.get("confidence", 0.8),
                "model_used": response.get("model_used", "unknown"),
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "tokens_used": response.get("tokens_used", {}),
                "cost": response.get("cost", 0.0)
            })

            # Match against known patterns
            matched_patterns = self._match_industry_patterns(analysis)
            analysis["matched_patterns"] = matched_patterns

            logger.info(f"Business analysis completed: {analysis.get('industry')} - {analysis.get('business_model')}")
            return analysis

        except Exception as e:
            logger.error(f"Business intent analysis failed: {e}")
            # Return basic fallback analysis
            return {
                "industry": "Technology & Software",
                "business_model": "SaaS",
                "core_value_proposition": "Software solution for business needs",
                "revenue_model": "Subscription-based",
                "confidence": 0.3,
                "error": str(e)
            }

    async def _extract_features(self, description: str, business_analysis: Dict[str, Any]) -> List[str]:
        """Extract detailed feature requirements from business description"""

        industry = business_analysis.get("industry", "Technology & Software")
        business_model = business_analysis.get("business_model", "SaaS")

        prompt = f"""
        Based on the business description and analysis, extract detailed feature requirements:

        Business Description: "{description}"
        Industry: {industry}
        Business Model: {business_model}

        Identify specific features needed for this application. Consider:

        1. Core Features (essential for basic functionality)
        2. User Management Features (if needed)
        3. Data Management Features
        4. Integration Features
        5. Reporting/Analytics Features
        6. Security Features
        7. Mobile/Responsive Features
        8. Industry-Specific Features

        For each feature, specify:
        - Feature name
        - Description
        - Priority (Critical, High, Medium, Low)
        - Complexity (Simple, Medium, Complex)
        - Dependencies on other features

        Return as a JSON list of feature objects.
        """

        try:
            response = await self.llm_router.route_request(
                prompt=prompt,
                task_type=TaskType.BUSINESS_INTENT,
                complexity=self.context.complexity_score,
                max_tokens=1500
            )

            content = response["content"]

            # Extract JSON from response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                try:
                    features = json.loads(json_match.group())
                    return [f["name"] if isinstance(f, dict) else str(f) for f in features]
                except json.JSONDecodeError:
                    pass

            # Fallback: extract features from text
            features = []
            lines = content.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['feature', 'functionality', 'capability']):
                    # Extract feature name from line
                    feature = re.sub(r'^[-*•]\s*', '', line.strip())
                    if feature and len(feature) < 100:
                        features.append(feature)

            return features[:20]  # Limit to 20 features

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return [
                "User authentication",
                "Core application functionality",
                "Data management",
                "User interface",
                "Basic reporting"
            ]

    async def _identify_constraints(self, business_analysis: Dict[str, Any], features: List[str]) -> List[Dict[str, Any]]:
        """Identify business, technical, and regulatory constraints"""

        industry = business_analysis.get("industry", "")
        business_model = business_analysis.get("business_model", "")

        constraints = []

        # Business constraints
        constraints.extend([
            {
                "type": "business",
                "description": "Budget limitations for MVP development",
                "priority": 1,
                "validation_rule": "cost <= budget_limit"
            },
            {
                "type": "timeline",
                "description": "Time-to-market pressure for competitive advantage",
                "priority": 2,
                "validation_rule": "development_time <= market_window"
            }
        ])

        # Industry-specific regulatory constraints
        regulatory_constraints = self._get_regulatory_constraints(industry)
        constraints.extend(regulatory_constraints)

        # Technical constraints based on complexity
        if self.context.complexity_score > 0.7:
            constraints.append({
                "type": "technical",
                "description": "High scalability requirements due to complexity",
                "priority": 1,
                "validation_rule": "architecture supports horizontal scaling"
            })

        # Feature-based constraints
        if any("payment" in f.lower() for f in features):
            constraints.append({
                "type": "regulatory",
                "description": "PCI DSS compliance for payment processing",
                "priority": 1,
                "validation_rule": "payment_system is PCI_DSS_compliant"
            })

        if any("user" in f.lower() and "data" in f.lower() for f in features):
            constraints.append({
                "type": "regulatory",
                "description": "GDPR compliance for user data handling",
                "priority": 1,
                "validation_rule": "data_handling is GDPR_compliant"
            })

        return constraints

    async def _generate_technical_specifications(
        self,
        business_analysis: Dict[str, Any],
        features: List[str],
        constraints: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comprehensive technical specifications"""

        prompt = f"""
        Generate technical specifications for the following application:

        Business Analysis: {json.dumps(business_analysis, indent=2)}
        Features Required: {json.dumps(features, indent=2)}
        Constraints: {json.dumps(constraints, indent=2)}

        Provide technical specifications including:

        1. Architecture Pattern:
           - Monolithic, Microservices, or Hybrid
           - Reasoning for choice

        2. Technology Stack Recommendations:
           - Frontend framework (React, Vue, Angular, etc.)
           - Backend framework (FastAPI, Express, Django, etc.)
           - Database (PostgreSQL, MongoDB, etc.)
           - Additional services needed

        3. Infrastructure Requirements:
           - Hosting approach (Cloud, On-premise, Hybrid)
           - Scalability considerations
           - Security requirements

        4. Integration Requirements:
           - Third-party APIs needed
           - External service dependencies
           - Data exchange formats

        5. Performance Requirements:
           - Expected user load
           - Response time targets
           - Throughput requirements

        6. Security Specifications:
           - Authentication method
           - Authorization approach
           - Data encryption needs

        Return as a structured JSON object with clear technical recommendations.
        """

        try:
            response = await self.llm_router.route_request(
                prompt=prompt,
                task_type=TaskType.ARCHITECTURE_DESIGN,
                complexity=self.context.complexity_score,
                max_tokens=2000
            )

            content = response["content"]

            # Try to parse JSON response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    specs = json.loads(json_match.group())
                    return specs
                except json.JSONDecodeError:
                    pass

            # Fallback specifications
            return {
                "architecture_pattern": "Monolithic" if self.context.complexity_score < 0.6 else "Microservices",
                "frontend_framework": "React with Next.js",
                "backend_framework": "Python with FastAPI",
                "database": "PostgreSQL",
                "hosting": "Cloud (Kubernetes)",
                "authentication": "JWT-based",
                "api_design": "RESTful APIs",
                "real_time_features": any("real-time" in f.lower() for f in features),
                "mobile_support": any("mobile" in f.lower() for f in features)
            }

        except Exception as e:
            logger.error(f"Technical specification generation failed: {e}")
            return {"error": str(e)}

    async def _create_project_roadmap(self, technical_specs: Dict[str, Any], complexity: float) -> Dict[str, Any]:
        """Create a development roadmap with timeline and milestones"""

        # Base development time estimation
        base_weeks = 4 + (complexity * 8)  # 4-12 weeks based on complexity

        phases = [
            {
                "phase": "Foundation",
                "duration_weeks": max(1, base_weeks * 0.2),
                "tasks": [
                    "Project setup and infrastructure",
                    "Core architecture implementation",
                    "Basic authentication system"
                ]
            },
            {
                "phase": "Core Development",
                "duration_weeks": max(2, base_weeks * 0.4),
                "tasks": [
                    "Core feature implementation",
                    "Database design and setup",
                    "API development"
                ]
            },
            {
                "phase": "Frontend Development",
                "duration_weeks": max(1, base_weeks * 0.25),
                "tasks": [
                    "UI/UX implementation",
                    "Frontend-backend integration",
                    "Responsive design"
                ]
            },
            {
                "phase": "Testing & Deployment",
                "duration_weeks": max(1, base_weeks * 0.15),
                "tasks": [
                    "Comprehensive testing",
                    "Performance optimization",
                    "Production deployment"
                ]
            }
        ]

        # Cost estimation (rough)
        cost_per_week = 2000  # Assuming automated generation costs
        total_cost = base_weeks * cost_per_week

        return {
            "phases": phases,
            "estimated_duration_weeks": base_weeks,
            "estimated_cost_usd": total_cost,
            "critical_path": ["Foundation", "Core Development"],
            "risk_factors": [
                "Third-party API dependencies" if technical_specs.get("integrations") else None,
                "Performance requirements" if complexity > 0.7 else None,
                "Regulatory compliance" if any("regulatory" in str(c) for c in technical_specs.values()) else None
            ]
        }

    def _load_industry_patterns(self) -> Dict[str, Any]:
        """Load industry-specific patterns and templates"""
        # This would typically load from a database or configuration file
        return {
            "E-commerce & Retail": {
                "common_features": ["product_catalog", "shopping_cart", "payment_processing", "order_management"],
                "required_integrations": ["payment_gateway", "shipping_api", "inventory_management"],
                "compliance_requirements": ["PCI_DSS", "consumer_protection"]
            },
            "Healthcare & Medical": {
                "common_features": ["patient_management", "appointment_scheduling", "medical_records"],
                "required_integrations": ["hl7_fhir", "insurance_apis"],
                "compliance_requirements": ["HIPAA", "FDA_regulations"]
            },
            "Financial Services": {
                "common_features": ["account_management", "transaction_processing", "reporting"],
                "required_integrations": ["banking_apis", "credit_scoring"],
                "compliance_requirements": ["PCI_DSS", "SOX", "KYC", "AML"]
            }
        }

    def _load_business_models(self) -> Dict[str, Any]:
        """Load business model templates and patterns"""
        return {
            "SaaS": {
                "revenue_streams": ["subscription", "usage_based", "tiered_pricing"],
                "key_metrics": ["MRR", "churn_rate", "LTV", "CAC"],
                "typical_features": ["user_management", "billing", "analytics", "integrations"]
            },
            "Marketplace": {
                "revenue_streams": ["commission", "listing_fees", "premium_features"],
                "key_metrics": ["GMV", "take_rate", "active_users", "transaction_volume"],
                "typical_features": ["user_matching", "payment_processing", "rating_system", "search"]
            }
        }

    def _load_compliance_rules(self) -> Dict[str, List[Dict[str, str]]]:
        """Load compliance requirements by industry"""
        return {
            "Healthcare & Medical": [
                {"name": "HIPAA", "description": "Health Insurance Portability and Accountability Act"},
                {"name": "FDA", "description": "Food and Drug Administration regulations"}
            ],
            "Financial Services": [
                {"name": "PCI_DSS", "description": "Payment Card Industry Data Security Standard"},
                {"name": "SOX", "description": "Sarbanes-Oxley Act compliance"},
                {"name": "KYC", "description": "Know Your Customer requirements"}
            ],
            "E-commerce & Retail": [
                {"name": "PCI_DSS", "description": "Payment Card Industry Data Security Standard"},
                {"name": "GDPR", "description": "General Data Protection Regulation"}
            ]
        }

    def _match_industry_patterns(self, analysis: Dict[str, Any]) -> List[str]:
        """Match business analysis against known industry patterns"""
        industry = analysis.get("industry", "")
        business_model = analysis.get("business_model", "")

        patterns = []

        if industry in self.industry_patterns:
            patterns.append(f"industry_pattern_{industry.lower().replace(' ', '_')}")

        if business_model in self.business_models:
            patterns.append(f"business_model_{business_model.lower()}")

        return patterns

    def _get_regulatory_constraints(self, industry: str) -> List[Dict[str, Any]]:
        """Get regulatory constraints for specific industry"""
        constraints = []

        if industry in self.compliance_rules:
            for rule in self.compliance_rules[industry]:
                constraints.append({
                    "type": "regulatory",
                    "description": f"{rule['name']}: {rule['description']}",
                    "priority": 1,
                    "validation_rule": f"system is {rule['name']}_compliant"
                })

        return constraints

    def _parse_business_analysis_fallback(self, text: str) -> Dict[str, Any]:
        """Fallback parser for business analysis when JSON parsing fails"""
        analysis = {}

        # Extract industry
        industry_match = re.search(r'industry[:\s]+([^\n]+)', text, re.IGNORECASE)
        if industry_match:
            analysis["industry"] = industry_match.group(1).strip()

        # Extract business model
        model_match = re.search(r'business model[:\s]+([^\n]+)', text, re.IGNORECASE)
        if model_match:
            analysis["business_model"] = model_match.group(1).strip()

        # Extract value proposition
        value_match = re.search(r'value proposition[:\s]+([^\n]+)', text, re.IGNORECASE)
        if value_match:
            analysis["core_value_proposition"] = value_match.group(1).strip()

        return analysis