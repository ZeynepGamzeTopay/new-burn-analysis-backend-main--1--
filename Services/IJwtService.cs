namespace BurnAnalysisApp.Services
{
    public interface IJwtService
    {
        string GenerateToken(int doctorId, string email);
    }
}
